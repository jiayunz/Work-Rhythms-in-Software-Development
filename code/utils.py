# encoding: utf-8
import json
from tqdm import tqdm
from datetime import datetime
import numpy as np
import re
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

data_dir = '../data'
raw_data_path = os.path.join(data_dir, 'data.json')
user_path = os.path.join(data_dir, 'users.json')
account2no_path = os.path.join(data_dir, 'account2no.json')
no2account_path = os.path.join(data_dir, 'no2account.json')
contributor_path = os.path.join(data_dir, 'contributor.json')
contributor_label_path = os.path.join(data_dir, 'user_label.json')
repo_label_path = os.path.join(data_dir, 'repo_label.json')
repo_path = os.path.join(data_dir, 'repo.json')
graph_path = os.path.join(data_dir, 'graph.txt')
shs_path = os.path.join(data_dir, 'shs.out')


def get_commit_activities(commits):
    commit_profile = np.zeros((7, 24))
    for c in commits:
        # commit without time zone
        if c['commit']['author']['date'][-1] == 'Z':
            #print 'No timezone information, skip.'
            continue
        date = datetime.strptime(c['commit']['author']['date'][:19], '%Y-%m-%dT%H:%M:%S')
        commit_profile[date.weekday()][date.hour] += 1

    return commit_profile


def get_h_index(stars):
    # start the h-index with the number of repos
    h_index = len(stars)
    while h_index > 0:
        count = 0
        # count: number of repos whose stars > ind
        for r_star in stars:
            if r_star >= h_index:
                count += 1
        if count >= h_index:
            break
        h_index -= 1

    return h_index


def get_cdf(quality_dict, cluster_dict):
    stats = {k: {l: {} for l in set(cluster_dict.values())} for k in quality_dict.values()[0]}

    for repo in set(quality_dict.keys()) & set(cluster_dict.keys()):
        for k in quality_dict[repo]:
            try:
                stats[k][cluster_dict[repo]][quality_dict[repo][k]] += 1
            except:
                stats[k][cluster_dict[repo]][quality_dict[repo][k]] = 1

    for k in stats:
        for l in stats[k]:
            data = sorted(stats[k][l].items(), key=lambda x:x[0])

            stats_k = [data[0][0]]
            stats_v = [1. * data[0][1] / sum(stats[k][l].values())]

            for k_val, v_val in data[1:]:
                stats_k.append(k_val)
                stats_v.append(stats_v[-1] + 1. * v_val / sum(stats[k][l].values()))

            print(k+'_k_'+str(l), '=', stats_k)
            print(k+'_v_'+str(l), '=', stats_v)

def get_pdf(quality_dict, cluster_dict):
    stats = {k: {l: {} for l in set(cluster_dict.values())} for k in quality_dict.values()[0]}

    for repo in set(quality_dict.keys()) & set(cluster_dict.keys()):
        for k in quality_dict[repo]:
            try:
                stats[k][cluster_dict[repo]][quality_dict[repo][k]] += 1
            except:
                stats[k][cluster_dict[repo]][quality_dict[repo][k]] = 1

    for k in stats:
        for l in stats[k]:
            data = sorted(stats[k][l].items(), key=lambda x:x[0])

            stats_k = []
            stats_v = []

            for k_val, v_val in data:
                stats_k.append(k_val)
                stats_v.append(1. * v_val / sum(stats[k][l].values()))

            print(k+'_k_'+str(l), '=', stats_k)
            print(k+'_v_'+str(l), '=', stats_v)


def get_box(quality_dict, cluster_dict):
    stats = {k: {l: [] for l in set(cluster_dict.values())} for k in list(quality_dict.values())[0]}
    data = []

    for u in set(quality_dict.keys()) & set(cluster_dict.keys()):
        d = {'pattern': '#'+str(cluster_dict[u])}
        for k in quality_dict[u]:
            stats[k][cluster_dict[u]].append(quality_dict[u][k])
            d[k] = quality_dict[u][k]
        data.append(d)

    data = pd.DataFrame(data)
    for k in stats:
        for l in stats[k]:
            print(k + '_' + str(l) + ' =', stats[k][l])

        plt.figure(figsize=(6, 2.5))
        sns.violinplot(x="pattern", y=k, data=data, palette="Set2", cut=0, scale='count')
        plt.ylabel(k, fontsize=12)
        plt.xlabel('')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.ylim(ymin=-1000, ymax=11000)
        plt.legend(loc=0)
        plt.savefig(k+'.eps', format='eps', dpi=1000)
        # plt.show()


def get_contributor_commits(data_rpath, account2no_rpath, data_wpath):
    sha = {}
    data = {}
    with open(account2no_rpath, 'r') as rf:
        account2no = json.load(rf)

    with open(data_rpath, 'r') as rf:
        lines = rf.readlines()
        for line in tqdm(lines, total=len(lines)):
            repo = json.loads(line)
            for c in repo['commits']:
                # avoid repeat commits
                if c['sha'] in sha:
                    continue
                else:
                    sha[c['sha']] = 0

                # if record timezone
                if c['commit']['author']['date'][-1] == 'Z':
                    continue

                if c['author'] == None:
                    author_id = None
                else:
                    author_id = str(c['author']['id'])

                # filter empty email
                if not re.search('@', c['commit']['author']['email']):
                    author_email = None
                else:
                    author_email = c['commit']['author']['email']

                if author_id:
                    author = account2no[str(author_id)]
                elif author_email:
                    author = account2no[author_email]
                else:
                    continue

                if author not in data:
                    data[author]= np.zeros((7, 24))

                d = datetime.strptime(c['commit']['author']['date'][:16], '%Y-%m-%dT%H:%M')
                data[author][d.weekday()][d.hour] += 1

    for u in data.keys():
        data[u] = data[u].tolist()

    with open(data_wpath, 'w') as wf:
        json.dump(data, wf, indent=4)

    return data

def get_repo_commits(data_rpath, data_wpath):
    sha = {}
    data = {}
    with open(data_rpath, 'r') as rf:
        lines = rf.readlines()
        for line in tqdm(lines, total=len(lines)):
            try:
                repo = json.loads(line)
                data[repo['full_name']] = np.zeros((7, 24))
                for c in repo['commits']:
                    # avoid repeat commits
                    if c['sha'] in sha:
                        continue
                    else:
                        sha[c['sha']] = 0

                    # if record timezone
                    if c['commit']['author']['date'][-1] == 'Z':
                        continue

                    d = datetime.strptime(c['commit']['author']['date'][:16], '%Y-%m-%dT%H:%M')
                    data[repo['full_name']][d.weekday()][d.hour] += 1

            except Exception as ex:
                print(ex)

    for repo in data.keys():
        data[repo] = data[repo].tolist()


    with open(data_wpath, 'w') as wf:
        json.dump(data, wf, indent=4)

    return data

def load_group(nohup_output):
    shs = []
    with open(nohup_output, 'r') as rf:
        for line in rf.readlines():
            if re.match('(.*?)\d+ th (.*?): ', line):
                i = re.sub('.*\d+ th (.*?): ', '', line.strip())
                shs.append(i)
    return shs

def cluster_demographic(data_rpath, label_rpath, account2no_rpath):
    with open(account2no_rpath, 'r') as rf:
        account2no = json.load(rf)

    with open(label_rpath, 'r') as rf:
        labels = json.load(rf)

    show_keys = ['blog', 'type', 'bio', 'company', 'name']
    with open(data_rpath, 'r') as rf:
        lines = rf.readlines()
        for line in tqdm(lines, total=len(lines)):
            u = json.loads(line)
            cls = labels[str(account2no[str(u['id'])])]['cluster']

            print(cls, {k: u[k] for k in show_keys})

def write_graph(data_rpath, no2account_wpath, account2no_wpath, graph_wpath):
    g = nx.Graph()
    with open(data_rpath, 'r') as rf:
        lines = rf.readlines()
        for line in tqdm(lines, total=len(lines)):
            repo = json.loads(line)

            for c in repo['commits']:
                if c['author'] == None:
                    author_id = None
                else:
                    author_id = str(c['author']['id'])
                if not re.search('@', c['commit']['author']['email']):
                    author_email = None
                else:
                    author_email = c['commit']['author']['email']

                if author_id and author_email:
                    g.add_edge(author_id, author_email)
                elif author_id:
                    g.add_node(author_id)
                elif author_email:
                    g.add_node(author_email)

    # the same connected component is the same user
    no2account = {}
    for i, comp in enumerate(nx.connected_components(g)):
        no2account[i] = list(comp)
    with open(no2account_wpath, 'w') as wf:
        json.dump(no2account, wf, indent=4)

    account2no = {}
    for no, accounts in no2account.items():
        for acc in accounts:
            account2no[acc] = no
    with open(account2no_wpath, 'w') as wf:
        json.dump(account2no, wf, indent=4)

    contributors = {}
    with open(data_rpath, 'r') as rf:
        lines = rf.readlines()
        for line in tqdm(lines, total=len(lines)):
            repo = json.loads(line)
            contributors[repo['full_name']] = []
            for c in repo['commits']:
                # if has time zone
                #if c['commit']['author']['date'][-1] == 'Z':
                #    continue
                if c['author'] == None:
                    author_id = None
                else:
                    author_id = str(c['author']['id'])

                # filter empty email
                if not re.search('@', c['commit']['author']['email']):
                    author_email = None
                else:
                    author_email = c['commit']['author']['email']

                if author_id:
                    contributors[repo['full_name']].append(account2no[author_id])
                elif author_email:
                    contributors[repo['full_name']].append(account2no[author_email])
                else:
                    continue

            contributors[repo['full_name']] = list(set(contributors[repo['full_name']]))

    edges = []

    for repo in tqdm(contributors, total=len(contributors)):
        for i, u in enumerate(contributors[repo]):
            for v in contributors[repo]:
                if u == v:
                    continue
                edges.append((u, v))

    print('# of nodes:', len(account2no))
    print('# of edges:', len(edges))

    with open(graph_wpath, 'w') as wf:
        wf.write(str(len(account2no)) + ' ' + str(len(edges)) + '\n')
        for u, v in edges:
            wf.write(str(u) + ' ' + str(v) + '\n')


if __name__ == '__main__':
    get_contributor_commits(raw_data_path, account2no_path, contributor_path)
    get_repo_commits(raw_data_path, repo_path)
    write_graph(raw_data_path, no2account_path, account2no_path, graph_path)

