# encoding: utf-8
from utils import *

current = datetime.strptime('2019-07-31 00:00:00', '%Y-%m-%d %H:%M:%S')

def get_quality(user):
    stars = []
    for repo in user['repos_list']:
        stars.append(repo['stargazers_count'])

    if not len(stars):
        popularity = {
            'followers': user['followers'],
            'age': (current - datetime.strptime(user['created_at'],'%Y-%m-%dT%H:%M:%SZ')).days,
            'avg_stars': 0,
            'h_index': 0
        }

    else:
         popularity = {
            'followers': user['followers'],
            'age': (current - datetime.strptime(user['created_at'], '%Y-%m-%dT%H:%M:%SZ')).days,
            'avg_stars': np.average(stars),
            'h_index': get_h_index(stars)
        }

    return popularity


def compare_user(data_rpath, label_rpath, account2no_rpath, method=None):
    with open(label_rpath, 'r') as rf:
        labels = json.load(rf)

    with open(account2no_rpath, 'r') as rf:
        account2no = json.load(rf)

    quality_dict = {}
    cluster_dict = {}
    with open(data_rpath, 'r') as rf:
        lines = rf.readlines()
        for line in tqdm(lines, total=len(lines)):
            u = json.loads(line)
            # get cluster id
            cluster_dict[account2no[str(u['id'])]] = labels[str(account2no[str(u['id'])])]['cluster']
            quality_dict[account2no[str(u['id'])]] = get_quality(u)

    if method == 'box':
        get_box(quality_dict, cluster_dict)
    elif method == 'cdf':
        get_cdf(quality_dict, cluster_dict)
    elif method == 'pdf':
        get_pdf(quality_dict, cluster_dict)


def compare_in_shs(data_rpath, label_rpath, shs_rpath, account2no_rpath):
    group = load_group(shs_rpath)

    with open(label_rpath, 'r') as rf:
        labels = json.load(rf)

    with open(account2no_rpath, 'r') as rf:
        account2no = json.load(rf)

    quality_dict = {}
    cluster_dict = {}
    with open(data_rpath, 'r') as rf:
        lines = rf.readlines()
        for line in tqdm(lines, total=len(lines)):
            u = json.loads(line)
            # get cluster id
            if str(account2no[str(u['id'])]) in group:
                cluster_dict[account2no[str(u['id'])]] = labels[str(account2no[str(u['id'])])]['cluster']
                quality_dict[account2no[str(u['id'])]] = get_quality(u)

    get_box(quality_dict, cluster_dict)


def compare_repo(data_rpath, label_rpath, method=None):
    with open(label_rpath, 'r') as rf:
        labels = json.load(rf)

    quality_dict = {}
    cluster_dict = {}
    with open(data_rpath, 'r') as rf:
        try:
            lines = rf.readlines()
            for line in tqdm(lines, total=len(lines)):
                repo = json.loads(line)
                # get cluster id
                if repo['full_name'] not in labels:
                    continue
                cluster_dict[repo['full_name']] = labels[repo['full_name']]['cluster']
                quality_dict[repo['full_name']] = {
                    'size': repo['size'],
                    'forks': repo['forks'],
                    'stars': repo['stargazers_count'],
                    'issues': repo['open_issues_count'],
                    'created_at': (current - datetime.strptime(repo['created_at'], '%Y-%m-%dT%H:%M:%SZ')).days,
                    'updated_at': (current - datetime.strptime(repo['updated_at'], '%Y-%m-%dT%H:%M:%SZ')).days,
                    'active_period': (datetime.strptime(repo['updated_at'], '%Y-%m-%dT%H:%M:%SZ') - datetime.strptime(repo['created_at'], '%Y-%m-%dT%H:%M:%SZ')).days,
                }
        except Exception as ex:
            print(ex)

    if method == 'box':
        get_box(quality_dict, cluster_dict)
    elif method == 'cdf':
        get_cdf(quality_dict, cluster_dict)
    elif method == 'pdf':
        get_pdf(quality_dict, cluster_dict)


def hourly_LOCadd(data_rpath, label_rpath):
    with open(label_rpath, 'r') as rf:
        labels = json.load(rf)

    data = {0: {'additions': [], 'deletions': []}, 1: {'additions': [], 'deletions': []}, 2: {'additions': [], 'deletions': []}}
    cluster_dict = {}
    quality_dict = {}

    with open(data_rpath, 'r') as rf:
        lines = rf.readlines()
        for line in tqdm(lines, total=len(lines)):
            repo = json.loads(line)
            if repo['full_name'] not in labels or not(len(repo['commits'])):
                continue
            cls = labels[repo['full_name']]['cluster']
            cluster_dict[repo['full_name']] = cls
            quality_dict[repo['full_name']] = {}
            commit_dates = [datetime.strptime(c['commit']['author']['date'][:16], '%Y-%m-%dT%H:%M') for c in
                            repo['commits']]
            first_date = min(commit_dates).date()
            last_date = max(commit_dates).date()
            #n_weeks = (last_date - first_date).days / 7

            d = {'additions': np.zeros(24), 'deletions': np.zeros(24)}
            commit_count = np.zeros(24)

            for c in repo['commits']:
                if c['commit']['author']['date'][-1] == 'Z':
                    # print 'No timezone information, skip.'
                    continue
                date = datetime.strptime(c['commit']['author']['date'][:19], '%Y-%m-%dT%H:%M:%S')
                # the last week could be incomplete. don't count
                #if (date.date() - first_date).days / 7 == n_weeks:
                #    continue
                d['additions'][date.hour] += c['stats']['additions']
                d['deletions'][date.hour] += c['stats']['deletions']
                commit_count[date.hour] += 1

            data[cls]['additions'].append(d['additions'] / (commit_count + 1e-26))
            data[cls]['deletions'].append(d['deletions'] / (commit_count + 1e-26))
            quality_dict[repo['full_name']] = {k+'_'+str(i): d[k][i] / (commit_count + 1e-26) for i in range(24) for k in d}

    for c in data:
        for k in data[c]:
            print(k+'_'+str(c)+' =', np.median(data[c][k], axis=0).tolist())

def LOC_add(data_rpath, label_rpath, method=None):
    with open(label_rpath, 'r') as rf:
        labels = json.load(rf)

    quality_dict = {}
    cluster_dict = {}

    with open(data_rpath, 'r') as rf:
        lines = rf.readlines()
        for line in tqdm(lines, total=len(lines)):
            try:
                repo = json.loads(line)
                if not (len(repo['commits'])) or repo['full_name'] not in labels:
                    continue
                commit_dates = [datetime.strptime(c['commit']['author']['date'][:16], '%Y-%m-%dT%H:%M') for c in
                                repo['commits']]

                commit_stats = [{
                    'additions': c['stats']['additions'],
                    'deletions': c['stats']['deletions'],
                    'total': c['stats']['total']
                } for c in repo['commits']]

                first_date = min(commit_dates).date()
                last_date = max(commit_dates).date()
                n_weeks = (last_date - first_date).days // 7 + 1
                # the last week could be incomplete. don't count
                weeks = [{'additions': 0, 'deletions': 0, 'total': 0} for _ in range(n_weeks)]
                for date, stats in zip(commit_dates, commit_stats):
                    w = (date.date() - first_date).days // 7
                    weeks[w]['additions'] += stats['additions']
                    weeks[w]['deletions'] += stats['deletions']
                    weeks[w]['total'] += stats['total']

                for w in range(n_weeks):
                    if weeks[w]['total'] == 0:
                        weeks[w]['additions'] = np.NaN
                        weeks[w]['deletions'] = np.NaN
                        weeks[w]['total'] = np.NaN

                cluster_dict[repo['full_name']] = labels[repo['full_name']]['cluster']
                quality_dict[repo['full_name']] = {
                    'additions': np.nanmedian([weeks[w]['additions'] for w in range(n_weeks - 1)]),
                    'deletions': np.nanmedian([weeks[w]['deletions'] for w in range(n_weeks - 1)]),
                    'total': np.nanmedian([weeks[w]['total'] for w in range(n_weeks - 1)])
                }
            except Exception as ex:
                print(ex)

    if method == 'box':
        get_box(quality_dict, cluster_dict)
    elif method == 'cdf':
        get_cdf(quality_dict, cluster_dict)


if __name__ == '__main__':
    data_dir = '../data'
    # user
    compare_user(user_path, contributor_label_path, account2no_path, method='box')
    compare_in_shs(user_path, contributor_label_path, shs_path, account2no_path)
    # repo
    compare_repo(raw_data_path, repo_label_path, method='box')
    # LOC
    LOC_add(raw_data_path, repo_label_path, method='box')
    hourly_LOCadd(raw_data_path, repo_label_path)
