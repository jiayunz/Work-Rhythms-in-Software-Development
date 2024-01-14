# encoding: utf-8
import matplotlib.pyplot as plt
from sklearn.cluster.bicluster import SpectralBiclustering
from utils import *
import pandas as pd


def get_features(commit_profiles):
    data = []
    for profile in tqdm(commit_profiles, total=len(commit_profiles)):
        profile = denoise(profile, 1)
        weekday = np.sum(profile[:5], axis=0) / np.sum(profile) / 5.
        weekend = np.sum(profile[5:], axis=0) / np.sum(profile) / 2.

        data.append(weekday.tolist() + weekend.tolist())

    return np.array(data)


def denoise(commit_profile, denoise=1):
    new_commit_profile = np.zeros(7*24)
    flat_commit_profile = np.reshape(commit_profile, (-1,))
    for i, c in enumerate(flat_commit_profile):
        avg_c = np.average([flat_commit_profile[(i+j)%(7*24)] for j in range(-denoise, denoise+1)])
        new_commit_profile[i] = avg_c

    new_commit_profile = np.reshape(new_commit_profile, (7, 24))
    #new_commit_profile = filters.gaussian_filter(commit_profile, sigma=0.5)
    return new_commit_profile


def biclustering(data, n_clusters, random_state=None):
    model = SpectralBiclustering(n_clusters=n_clusters, random_state=random_state)
    model.fit(data)
    print('# of row clusters:', {i: model.row_labels_.tolist().count(i) for i in range(len(set(model.row_labels_)))})
    print('share of row clusters:', {i: 1. * model.row_labels_.tolist().count(i) / len(data) for i in range(len(set(model.row_labels_)))})
    print('# of column clusters:', len(set(model.column_labels_)))
    print(model.column_labels_[:24])
    print(model.column_labels_[24:])

    return model.row_labels_, model.column_labels_

def plot_heatmap(data, title, cmap_no):
    data = pd.DataFrame(data, index=['Mon.', 'Tues.', 'Wed.', 'Thur.', 'Fri.', 'Sat.', 'Sun.'])
    plt.figure(figsize=(10, 3))
    cmap = ['RdPu', 'YlOrRd', 'YlGnBu', 'plasma_r', 'rocket_r', 'viridis_r', 'Wistia', 'summer_r']
    #plt.title(company)
    sns.heatmap(data, linewidths=0.01, square=True, xticklabels=3, yticklabels=2, linecolor='white', cmap=cmap[cmap_no], robust=False, vmin=0, vmax=0.025, cbar=True)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18, rotation='0')
    plt.xlabel('Hour of Day', fontsize=25)
    plt.ylabel('Day of Week', fontsize=25)

    plt.tight_layout()
    plt.title(title)
    #sav_fig = plt.gcf()  # 'get current figure'
    #sav_fig.savefig(title+'.eps', format='eps', dpi=1000)
    plt.show()


def contributor_clustering(k, data_rpath, no2account_rpath, label_wpath, min_commits=100):
    with open(no2account_rpath, 'r') as rf:
        no2account = json.load(rf)

    with open(data_rpath, 'r') as rf:
        contributor_data = json.load(rf)

    total = 0
    commit_profiles = {}
    for u in contributor_data.keys():
        if np.sum(contributor_data[u]) >= min_commits:
            total += np.sum(contributor_data[u])
            commit_profiles[u] = contributor_data[u]

    data = get_features(commit_profiles.values())

    row_labels_, col_labels_ = biclustering(data, n_clusters=k, random_state=1333)
    #fit_data = data[np.argsort(row_labels_)]
    #plot_heatmap(fit_data, str(k)+'_result', cmap_no=0)

    clusters = {c: np.zeros(48) for c in set(row_labels_)}
    data_array = np.array(data)
    for i, d in enumerate(data_array):
        clusters[row_labels_[i]] += d

    labels = {}
    for no, l in zip(commit_profiles.keys(), row_labels_):
        labels[no] = {
            'accounts': no2account[no],
            'cluster': int(l)
        }

    with open(label_wpath, 'w') as wf:
        json.dump(labels, wf, indent=4)

    cluster_heatmap(data_rpath, label_wpath)

    return labels


def repo_clustering(k, data_rpath, label_wpath, min_commits=100):
    with open(data_rpath, 'r') as rf:
        commit_profiles = json.load(rf)

    profile_keys = set(commit_profiles.keys())
    for u in profile_keys:
        if np.sum(commit_profiles[u]) < min_commits:
            del commit_profiles[u]

    data = get_features(commit_profiles.values())

    row_labels_, col_labels_ = biclustering(data, n_clusters=k, random_state=1333)

    clusters = {c: np.zeros(48) for c in set(row_labels_)}
    data_array = np.array(data)
    for i, d in enumerate(data_array):
        clusters[row_labels_[i]] += d

    plt.figure(figsize=(9, 3))
    plt.subplot(1,2,1)
    for c in clusters:
        cluster_data = (clusters[c] / row_labels_.tolist().count(c)).tolist()
        plt.plot(cluster_data[:24] + cluster_data[0:1], label=f'Pattern #{c}')
        print('cluster'+str(c), '=', cluster_data)
    plt.ylim([0, 0.024])
    plt.xlim([0, 24])
    plt.xlabel('Hour of Day')
    plt.ylabel('Commit Frequency')
    plt.title('Weekday')
    plt.legend()

    plt.subplot(1, 2, 2)
    for c in clusters:
        cluster_data = (clusters[c] / row_labels_.tolist().count(c)).tolist()
        plt.plot(cluster_data[24:] + cluster_data[24:25], label=f'Pattern #{c}')
        print('cluster' + str(c), '=', cluster_data)
    plt.ylim([0, 0.024])
    plt.xlim([0, 24])
    plt.xlabel('Hour of Day')
    plt.ylabel('Commit Frequency')
    plt.title('Weekend')
    plt.legend()
    plt.tight_layout()
    plt.show()

    labels = {}
    for repo, l in zip(commit_profiles, row_labels_):
        labels[repo] = {
            'accounts': repo,
            'cluster': int(l)
        }

    with open(label_wpath, 'w') as wf:
        json.dump(labels, wf, indent=4)

    return labels


def cluster_heatmap(data_rpath, label_rpath):
    with open(data_rpath, 'r') as rf:
        data = json.load(rf)

    with open(label_rpath, 'r') as rf:
        labels = json.load(rf)

    clusters = {labels[uid]['cluster']: [] for uid in labels}

    for u in data:
        if u not in labels:
            continue
        cls = labels[u]['cluster']
        clusters[cls].append(np.array(data[u]) / np.sum(data[u]))

    for cls in clusters:
        clusters[cls] = np.average(clusters[cls], axis=0)
        plot_heatmap(clusters[cls], 'Pattern #' + str(cls), 4)

if __name__ == '__main__':
    contributor_clustering(k=4, min_commits=30, data_rpath=contributor_path, no2account_rpath=no2account_path, label_wpath=contributor_label_path)
    repo_clustering(k=3, min_commits=300, data_rpath=repo_path, label_wpath=repo_label_path)

