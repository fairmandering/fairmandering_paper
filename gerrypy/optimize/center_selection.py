import random
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from gerrypy.utils.spatial_utils import *


def kmeans_initialization(capacities, state_df, n_random, init='++ prob'):
    k = len(capacities)
    n_blocks = len(state_df)
    locations = state_df[['x', 'y']].values

    if init == 'random':
        centers = np.random.choice(n_blocks, k, replace=False)
    else:
        # init k centers using kmeans++ initialization variant
        seed = np.random.choice(n_blocks)
        centers = [seed]
        first_round = True
        while len(centers) < k:
            min_sq_dist = ((locations[np.newaxis] -
                             locations[centers, np.newaxis]) ** 2)\
                .sum(axis=-1).min(axis=0)

            if init == '++ furthest':
                seed = np.argmax(min_sq_dist)
            elif init == '++ prob':
                seed = np.random.choice(n_blocks,
                                        p=min_sq_dist / np.sum(min_sq_dist))
            centers.append(seed)

            if first_round:
                centers = centers[1:]
                first_round = False

    return centers


def capacitated_kmeans(capacities, state_df, n_random=0, init='random'):
    k = len(capacities)
    n_blocks = len(state_df)
    locations = state_df[['x', 'y']].values

    centers = kmeans_initialization(capacities, state_df, n_random, init)
    print(centers)
    # initialize other data structures
    prev_centers = set()

    random_mask = np.ones(k)
    random_mask[np.random.choice(k, n_random, replace=False)] = 0
    all_centers = [centers]

    while set(centers) != prev_centers:
        # calculate distances
        dists = cdist(locations, locations[centers])
        search_order = np.argsort(dists.min(axis=1))

        fill_factor = np.ones(k)

        center_assignment = [[] for _ in range(k)]

        # iterate over precincts and update priorities
        for precinct_id in search_order:
            priority = dists[precinct_id] * fill_factor ** 2
            selected_center_ix = np.argmin(priority)
            center_assignment[selected_center_ix].append(precinct_id)

            fill_adjustment = state_df.loc[precinct_id, 'population'] / capacities[selected_center_ix]
            fill_factor[selected_center_ix] += fill_adjustment

        # calulate new centers
        centroid_positions = np.array([
            np.average(locations[precinct_ids], axis=0,
                       weights=state_df.loc[precinct_ids].population.values)
            for precinct_ids in center_assignment
        ])
        # print(centroid_positions)
        print(fill_factor - 1)

        centroid_dists = cdist(locations, centroid_positions)

        prev_centers = set(centers)
        centers = list(np.argmin(centroid_dists, axis=0))
        all_centers.append(centers)

    return all_centers


def kmeans_seeds(state_df, n_distrs, random_seeds=0, init='random'):

    weights = state_df.population.values + 1

    weights = rand_weight(random_seeds, weights)

    pts = state_df[['x', 'y']]

    kmeans = KMeans(n_clusters=n_distrs, init=init, n_jobs=-1) \
        .fit(pts, sample_weight=weights).cluster_centers_

    dists = cdist(kmeans, pts)
    centers = [state_df.index[i].item()  # Convert to native int for jsonability
               for i in list(np.argmin(dists, axis=1))]

    return centers


def rand_weight(n_seeds, weights):
    total_weight = weights.sum()
    for _ in range(n_seeds):
        rand_seed = random.randint(0, len(weights) - 1)
        weights[rand_seed] = total_weight
    return weights
