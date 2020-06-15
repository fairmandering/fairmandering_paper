import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering
from gerrypy.utils.spatial_utils import vecdist

# TODO: Move to baseline

def spectral_cluster(config, xij_activation):
    n_districts = config['n_districts']


    B = nx.Graph(xij_activation)  # Bipartite graph
    A = nx.adjacency_matrix(B, weight='weight').toarray()
    sc = SpectralClustering(n_districts, affinity='precomputed',
                            n_init=10, n_jobs=-1)
    clustering = sc.fit(A)
    labels = clustering.labels_

    # Post process clusters
    cluster_map = {n: l for n, l in zip(list(B.nodes), labels)}
    n_clusters = config['n_districts']
    cluster_ys = {i: [] for i in range(n_clusters)}
    cluster_xs = {i: [] for i in range(n_clusters)}
    for node, cluster in cluster_map.items():
        if node[0:6] == 'center':
            cluster_ys[cluster].append(int(node[6:]))
        else:
            cluster_xs[cluster].append(int(node))

    return cluster_ys


def select_centers(state_df, cluster_ys, y_activation, method='sample'):
    tracts = list(state_df.index)
    centers = []
    for i, cluster in cluster_ys.items():
        cluster_weights = np.array([y_activation[y] for y in cluster])
        cluster_weights = cluster_weights / sum(cluster_weights.flatten())
        cluster_positions = state_df.loc[cluster][['x', 'y']].values
        if method == 'average':
            cluster_center = cluster_weights.dot(cluster_positions).flatten()
            pdist = vecdist(cluster_center[1], cluster_center[0],
                            state_df['y'].values,
                            state_df['x'].values)
            center = np.argmin(pdist)
            centers.append(tracts[center])
        elif method == 'sample':
            center = np.random.choice(cluster, p=cluster_weights)
            centers.append(center)

    return centers
