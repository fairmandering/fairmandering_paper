import random
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from gerrypy.utils.spatial_utils import *


def geographic_kmeans_seeds(config, state_df, random_seeds=0, init='random'):
    n_distrs = config['n_districts']
    weights = state_df['population'].values + 1

    weights = rand_weight(random_seeds, weights)

    pts = state_df[['x', 'y']]
    pts = geo_to_euclidean_coords(pts[:, 1], pts[:, 0])
    kmeans = KMeans(n_clusters=n_distrs, init=init, n_jobs=-1) \
        .fit(pts, sample_weight=weights).cluster_centers_
    kmeans = euclidean_coords_to_geo(kmeans[:, 0], kmeans[:, 1], kmeans[:, 2])
    centers = []
    for mean in kmeans:
        pdist = vecdist(mean[1], mean[0], pts[:, 1], pts[:, 0])
        center = np.argmin(pdist)
        centers.append(center)
    return centers


def euclidean_kmeans_seeds(config, state_df, random_seeds=0, init='random'):
    n_distrs = config['n_districts']
    weights = state_df.population.values + 1

    weights = rand_weight(random_seeds, weights)

    pts = state_df[['x', 'y']]

    kmeans = KMeans(n_clusters=n_distrs, init=init, n_jobs=-1) \
        .fit(pts, sample_weight=weights).cluster_centers_

    dists = cdist(kmeans, pts)
    centers = [state_df.index[i] for i in list(np.argmin(dists, axis=1))]

    return centers


def rand_weight(n_seeds, weights):
    total_weight = weights.sum()
    for _ in range(n_seeds):
        rand_seed = random.randint(0, len(weights) - 1)
        weights[rand_seed] = total_weight
    return weights
