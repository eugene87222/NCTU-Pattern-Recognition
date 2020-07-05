import numpy as np
from numpy.linalg import norm
from datetime import datetime
import matplotlib.pyplot as plt
from queue import Queue, PriorityQueue
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

CORE = 100
EDGE = 200
INF = 1e5
ENCODE = 1
MAPPING = {}
THRES = []
VERTICAL = []


def dbscan(points, radius, density_thres):
    point_num = points.shape[0]
    point_labels = np.zeros(points.shape[0])
    point_neighbors = []
    core = []
    non_core = []

    for i in range(points.shape[0]):
        neighbors = [j for j in range(points.shape[0]) if norm(points[i]-points[j])<=radius]
        if len(neighbors) >= density_thres:
            point_labels[i] = CORE
            core.append(i)
        else:
            non_core.append(i)
        point_neighbors.append(neighbors)

    for i in non_core:
        for neighbor in point_neighbors[i]:
            if neighbor in core:
                point_labels[i] = EDGE
                break

    cluster_idx = 1
    for i, label in enumerate(point_labels):
        q = Queue()
        if label == CORE:
            q.put(i)
            while not q.empty():
                for neighbor in point_neighbors[q.get()]:
                    if point_labels[neighbor] in [CORE, EDGE]:
                        if point_labels[neighbor] == CORE:
                            q.put(neighbor)
                        point_labels[neighbor] = cluster_idx
            cluster_idx += 1

    return point_labels


def plot_clusters(points, point_labels, cluster_idx, file=None):
    colors = [
        'red', 'magenta', 'lawngreen', 'darkorange', 'dodgerblue',
        'gold', 'navy', 'blueviolet', 
    ]
    if cluster_idx is None:
        clusters = np.unique(point_labels).astype(np.int32)
        labels = np.asarray(point_labels)
        has_outlier = 0
        for idx in clusters:
            if idx == 0:
                color = 'black'
                has_outlier = 1
            else:
                color = colors[idx%8]
            c_i = points[labels==idx]
            plt.scatter(c_i[:, 0], c_i[:, 1], c=color, s=15)
        cluster_num = clusters.shape[0] - has_outlier
        if file is not None:
            print(f'{cluster_num} clusters found', file=file)
            print(f'{np.sum(labels==0)} outliers', file=file)
        else:
            print(f'{cluster_num} clusters found')
            print(f'{np.sum(labels==0)} outliers')
    else:
        for i, c in enumerate(cluster_idx):
            color = colors[i%8]
            c_i = points[np.asarray(c)]
            plt.scatter(c_i[:, 0], c_i[:, 1], c=color, s=15)


def get_members(cluster):
    q = Queue()
    q.put(cluster)
    member = []
    while not q.empty():
        c = q.get()
        for m in c:
            if type(m) == list:
                q.put(m)
            elif type(m) == tuple:
                member.append(m[0])
    return member


def single_link(cluster_1, cluster_2, dis_mat):
    members_c1 = get_members(cluster_1)
    members_c2 = get_members(cluster_2)
    min_dis = INF
    for m_c1 in members_c1:
        for m_c2 in members_c2:
            dis = dis_mat[m_c1, m_c2]
            min_dis = min(dis, min_dis)
    return min_dis


def complete_link(cluster_1, cluster_2, dis_mat):
    members_c1 = get_members(cluster_1)
    members_c2 = get_members(cluster_2)
    max_dis = 0
    for m_c1 in members_c1:
        for m_c2 in members_c2:
            dis = dis_mat[m_c1, m_c2]
            max_dis = max(dis, max_dis)
    return max_dis


def average_link(cluster_1, cluster_2, dis_mat):
    members_c1 = get_members(cluster_1)
    members_c2 = get_members(cluster_2)
    dis = 0
    for m_c1 in members_c1:
        for m_c2 in members_c2:
            dis += dis_mat[m_c1, m_c2]
    return dis / (len(members_c1)*len(members_c2))


def distance(cluster_1, cluster_2, dis_mat, method):
    if method == 'single':
        return single_link(cluster_1, cluster_2, dis_mat)
    elif method == 'complete':
        return complete_link(cluster_1, cluster_2, dis_mat)
    elif method == 'average':
        return average_link(cluster_1, cluster_2, dis_mat)


def get_mis_dis(clusters, dis_mat, method):
    n = len(clusters)
    min_dis = INF
    min_dis_pair = None
    for i in range(n):
        for j in range(i, n):
            if i == j:
                continue
            else:
                dis = distance(clusters[i], clusters[j], dis_mat, method)
                if dis < min_dis:
                    min_dis = dis
                    min_dis_pair = (i, j)
    return min_dis_pair, min_dis


def clustering(clusters, dis_mat, method):
    global THRES, VERTICAL
    merge_pair, min_dis = get_mis_dis(clusters, dis_mat, method)
    THRES.append(min_dis)
    VERTICAL.append((min_dis, clusters[merge_pair[0]][0][-1]))
    VERTICAL.append((min_dis, clusters[merge_pair[1]][0][-1]))
    target = clusters.pop(merge_pair[1])
    clusters[merge_pair[0]].append(target[0])
    clusters[merge_pair[0]] = [[clusters[merge_pair[0]], min_dis]]


def hc(clusters, dis_mat, method):
    cluster_num = len(clusters)
    while cluster_num > 1:
        clustering(clusters, dis_mat, method)
        cluster_num = len(clusters)
    return clusters


def data_mapping(clusters):
    global ENCODE, MAPPING
    _clusters = clusters[0]
    if type(_clusters) == list:
        data_mapping(_clusters[0])
        data_mapping(_clusters[1])
    else:
        MAPPING[str(_clusters[0])] = ENCODE
        ENCODE += 1


def get_center(clusters):
    _clusters = clusters[0]
    if type(_clusters) == list:
        left_center = get_center(_clusters[0])
        right_center = get_center(_clusters[1])
        return (left_center+right_center) / 2
    else:
        return MAPPING[str(_clusters[0])]


def draw_dendogram(clusters):
    _clusters = clusters[0]
    if type(_clusters) == list:
        left_center = get_center(_clusters[0])
        right_center = get_center(_clusters[1])
        center = (left_center+right_center) / 2
        plt.plot([left_center, right_center], [clusters[1], clusters[1]], c='blue', lw=2)
        plt.plot([left_center, left_center], [_clusters[0][1], clusters[1]], c='blue', lw=2)
        plt.plot([right_center, right_center], [_clusters[1][1], clusters[1]], c='blue', lw=2)
        draw_dendogram(_clusters[0])
        draw_dendogram(_clusters[1])


def get_opt_thres():
    global THRES
    THRES.append(0)
    THRES = sorted(THRES, reverse=True)
    max_gap = 0
    opt_thres = None
    for i in range(1, len(THRES)):
        gap = THRES[i-1] - THRES[i]
        if gap > max_gap:
            max_gap = gap
            opt_thres = (THRES[i-1]+THRES[i]) / 2
    return opt_thres


def get_opt_clusters(clusters_idx, clusters, thres):
    _clusters = clusters[0]
    if type(_clusters) == list:
        upper_bound = clusters[1]
        left_lower_bound = _clusters[0][1]
        right_lower_bound = _clusters[1][1]
        if thres<upper_bound and thres>left_lower_bound:
            clusters_idx.append(get_members(_clusters[0]))
        else:
            get_opt_clusters(clusters_idx, _clusters[0], thres)
        if thres<upper_bound and thres>right_lower_bound:
            clusters_idx.append(get_members(_clusters[1]))
        else:
            get_opt_clusters(clusters_idx, _clusters[1], thres)


def find_leader(clusters, idx):
    leader = idx
    while True:
        if type(clusters[leader]) == int:
            leader = clusters[leader]
        else:
            return leader


def mst(clusters, dis_mat):
    global THRES, VERTICAL
    q = PriorityQueue()
    for i in range(0, dis_mat.shape[0]):
        for j in range(i+1, dis_mat.shape[1]):
            q.put((dis_mat[i, j], (i, j)))
    cluster_num = len(clusters)
    label = np.arange(cluster_num)
    while cluster_num > 1:
        merge_pair = q.get()
        dis = merge_pair[0]
        pair = merge_pair[1]
        leader_0 = find_leader(clusters, pair[0])
        leader_1 = find_leader(clusters, pair[1])
        if label[leader_0] == label[leader_1]:
            continue
        label[leader_1] = leader_0
        THRES.append(dis)
        VERTICAL.append((dis, clusters[leader_0][0][-1]))
        VERTICAL.append((dis, clusters[leader_1][0][-1]))
        clusters[leader_0].append(clusters[leader_1][0])
        clusters[leader_0] = [[clusters[leader_0], dis]]
        clusters[leader_1] = leader_0
        cluster_num -= 1
    for t in clusters:
        if type(t) != int:
            return [t]


def complete_hc(clusters, dis_mat):
    global THRES, VERTICAL
    dis_list = []
    edge_map = {}
    for i in range(dis_mat.shape[0]):
        for j in range(i, dis_mat.shape[1]):
            if i == j:
                continue
            dis_list.append((dis_mat[i, j], (i, j)))
            edge_map[(i, j)] = True
            edge_map[(j, i)] = True
    dis_list = sorted(dis_list, key=lambda t: t[0], reverse=True)
    cluster_num = len(clusters)
    label = np.arange(cluster_num)
    while cluster_num > 1:
        (dis, pair) = dis_list.pop()
        if not edge_map[pair]:
            continue
        leader_0 = find_leader(clusters, pair[0])
        leader_1 = find_leader(clusters, pair[1])
        if label[leader_0] == label[leader_1]:
            continue
        label[leader_1] = leader_0
        THRES.append(dis)
        VERTICAL.append((dis, clusters[leader_0][0][-1]))
        VERTICAL.append((dis, clusters[leader_1][0][-1]))
        clusters[leader_0].append(clusters[leader_1][0])
        clusters[leader_0] = [[clusters[leader_0], dis]]
        clusters[leader_1] = leader_0
        cluster_num -= 1

        merge_members = get_members(clusters[leader_0])
        other_clusters = []
        check = {}
        for i in range(len(label)):
            leader = find_leader(clusters, i)
            if leader_0!=leader and leader_1!=leader:
                if check.get(leader):
                    continue
                other_clusters.append(get_members(clusters[leader]))
                check[i] = True
        for cluster_members in other_clusters:
            dis_block = dis_mat[np.array(cluster_members), :]
            dis_block = dis_block[:, np.array(merge_members)]
            max_pos = np.unravel_index(np.argmax(dis_block), dis_block.shape)
            for m1 in cluster_members:
                for m2 in merge_members:
                    edge_map[(m1, m2)] = False
                    edge_map[(m2, m1)] = False
            edge_map[(cluster_members[max_pos[0]], merge_members[max_pos[1]])] = True
            edge_map[(merge_members[max_pos[1]], cluster_members[max_pos[0]])] = True
    for t in clusters:
        if type(t) != int:
            return [t]


def plot_ground_truth(points, y):
    colors = [
        'red', 'blue', 'gold', 'green', 'purple'
    ]
    labels = np.unique(y)
    for l in labels:
        p = points[y==l]
        plt.scatter(p[:, 0], p[:, 1], c=colors[l%5])


def close_fig(filename=None):
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=300, transparent=True)
    plt.clf()


def initialize_hc():
    global ENCODE, MAPPING, THRES, VERTICAL
    ENCODE = 1
    MAPPING = {}
    THRES = []
    VERTICAL = []


if __name__ == '__main__':
    DBSCAN = True
    HC = False
    if DBSCAN:
        for shape in ['circle', 'moon', 'varied']:
            for point_num in [100, 200]:
                if shape=='circle' or shape=='moon':
                    for noise in [0, 0.05, 0.1]:
                        points = np.load(f'./data/{shape}_{point_num}_{noise}.npy')
                        y = np.load(f'./data/{shape}_{point_num}_{noise}_y.npy')
                        plot_ground_truth(points, y)
                        close_fig(f'./dbscan/{shape}_{point_num}_{noise}_gt.png')
                        if shape == 'circle':
                            r = [0.1, 0.2, 0.3, 0.4, 0.5]
                        else:
                            r = [0.1, 0.3, 0.5, 0.7]
                        for radius in r:
                            for density_thres in [5, 10, 15]:
                                start = datetime.now()
                                log = open('dbscan.log', 'a+')
                                print(f'{shape}, {point_num} data points, noise={noise}, radius={radius}, density_thres={density_thres}', file=log)

                                point_labels = dbscan(points, radius, density_thres)
                                plot_clusters(points, point_labels, None, log)
                                close_fig(f'./dbscan/{shape}_{point_num}_{noise}_{radius}_{density_thres}.png')

                                y_pred = np.arange(point_num)
                                for i, l in enumerate(point_labels):
                                    y_pred[i] = -1 if l==0 else l - 1
                                ari_score = adjusted_rand_score(y, y_pred)

                                print(f'ARI score: {ari_score}', file=log)
                                print(f'spend {datetime.now() - start}', file=log)
                                print('====================', file=log)
                                log.close()
                else:
                    points = np.load(f'./data/{shape}_{point_num}.npy')
                    y = np.load(f'./data/{shape}_{point_num}_y.npy')
                    plot_ground_truth(points, y)
                    close_fig(f'./dbscan/{shape}_{point_num}_gt.png')
                    for radius in [1, 2, 3, 4]:
                        for density_thres in [5, 10, 15]:
                            start = datetime.now()
                            log = open('dbscan.log', 'a+')
                            print(f'{shape}, {point_num} data points, radius={radius}, density_thres={density_thres}', file=log)
                            point_labels = dbscan(points, radius, density_thres)
                            plot_clusters(points, point_labels, None, log)
                            close_fig(f'./dbscan/{shape}_{point_num}_{radius}_{density_thres}.png')

                            y_pred = np.arange(point_num)
                            for i, l in enumerate(point_labels):
                                y_pred[i] = -1 if l==0 else l - 1
                            ari_score = adjusted_rand_score(y, y_pred)

                            print(f'ARI score: {ari_score}', file=log)
                            print(f'spend {datetime.now() - start}', file=log)
                            print('====================', file=log)
                            log.close()
    if HC:
        for shape in ['circle', 'moon', 'varied']:
            for point_num in [100, 200]:
                if shape=='circle' or shape=='moon':
                    for noise in [0, 0.05, 0.1]:
                        points = np.load(f'./data/{shape}_{point_num}_{noise}.npy')
                        y = np.load(f'./data/{shape}_{point_num}_{noise}_y.npy')
                        plot_ground_truth(points, y)
                        close_fig(f'./{shape}/{shape}_{point_num}_{noise}_gt.png')
                        dis_mat = cdist(points, points)
                        for method in ['single', 'complete', 'average']:
                            initialize_hc()
                            clusters = [[[(i, points[i].tolist()), 0]] for i in range(points.shape[0])]
                            start = datetime.now()
                            if method == 'single':
                                clusters = mst(clusters, dis_mat)
                            elif method == 'complete':
                                clusters = complete_hc(clusters, dis_mat)
                            else:
                                clusters = hc(clusters, dis_mat, method)
                            data_mapping(clusters[0][0])

                            opt_thres = get_opt_thres()
                            draw_dendogram(clusters[0][0])
                            plt.plot([0, points.shape[0]+1], [opt_thres, opt_thres], c='red', ls='--', lw=3)
                            close_fig(f'./{shape}/{method}_{shape}_{point_num}_{noise}_dendogram.png')

                            opt_cluster_num = sum([1 if v[0]>opt_thres and v[1]<opt_thres else 0 for v in VERTICAL])
                            clusters_idx = []
                            get_opt_clusters(clusters_idx, clusters[0][0], opt_thres)
                            plot_clusters(points, None, clusters_idx)
                            close_fig(f'./{shape}/{method}_{shape}_{point_num}_{noise}.png')

                            y_pred = np.zeros(point_num, dtype=np.int32)
                            for i, cluster in enumerate(clusters_idx):
                                for member in cluster:
                                    y_pred[member] = i
                            ari_score = adjusted_rand_score(y, y_pred)
                            
                            log = open(f'{shape}.log', 'a+')
                            print(f'{shape}, {point_num} data points, noise={noise}, {method} link', file=log)
                            print(f'{opt_cluster_num} clusters found', file=log)
                            print(f'ARI score: {ari_score}', file=log)
                            print(f'spend {datetime.now() - start}', file=log)
                            print('====================', file=log)
                            log.close()
                else:
                    points = np.load(f'./data/{shape}_{point_num}.npy')
                    y = np.load(f'./data/{shape}_{point_num}_y.npy')
                    plot_ground_truth(points, y)
                    close_fig(f'./{shape}/{shape}_{point_num}_gt.png')
                    dis_mat = cdist(points, points)
                    for method in ['single', 'complete', 'average']:
                        initialize_hc()
                        clusters = [[[(i, points[i].tolist()), 0]] for i in range(points.shape[0])]
                        start = datetime.now()
                        if method == 'single':
                            clusters = mst(clusters, dis_mat)
                        elif method == 'complete':
                            clusters = complete_hc(clusters, dis_mat)
                        else:
                            clusters = hc(clusters, dis_mat, method)
                        data_mapping(clusters[0][0])

                        opt_thres = get_opt_thres()
                        draw_dendogram(clusters[0][0])
                        plt.plot([0, points.shape[0]+1], [opt_thres, opt_thres], c='red', ls='--', lw=3)
                        close_fig(f'./{shape}/{method}_{shape}_{point_num}_dendogram.png')

                        opt_cluster_num = sum([1 if v[0]>opt_thres and v[1]<opt_thres else 0 for v in VERTICAL])
                        clusters_idx = []
                        get_opt_clusters(clusters_idx, clusters[0][0], opt_thres)
                        plot_clusters(points, None, clusters_idx)
                        close_fig(f'./{shape}/{method}_{shape}_{point_num}.png')

                        y_pred = np.zeros(point_num, dtype=np.int32)
                        for i, cluster in enumerate(clusters_idx):
                            for member in cluster:
                                y_pred[member] = i
                        ari_score = adjusted_rand_score(y, y_pred)
                        
                        log = open(f'{shape}.log', 'a+')
                        print(f'{shape}, {point_num} data points, {method} link', file=log)
                        print(f'{opt_cluster_num} clusters found', file=log)
                        print(f'ARI score: {ari_score}', file=log)
                        print(f'spend {datetime.now() - start}', file=log)
                        print('====================', file=log)
                        log.close()
