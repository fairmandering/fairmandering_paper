import math
import random
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from gerrypy.utils.spatial_utils import *


def iterative_random(area_df, capacities, pdists):
    unassigned_blocks = list(area_df.index)
    np.random.shuffle(capacities)

    centers = []
    child_ix = 0
    block_seed = random.choice(unassigned_blocks)
    while child_ix < len(capacities):
        block_seed_sq_dist = pdists[block_seed, unassigned_blocks] ** 2
        center_p = block_seed_sq_dist / np.sum(block_seed_sq_dist)
        center_seed = np.random.choice(unassigned_blocks, p=center_p)

        if child_ix < len(capacities) - 1:
            assignment_order = np.argsort(pdists[center_seed, unassigned_blocks])

            blocks_assigned_to_center = []
            population_assigned_to_center = 0
            target_population = capacities[child_ix]
            assignment_ix = 0
            while population_assigned_to_center < target_population:
                block = unassigned_blocks[assignment_order[assignment_ix]]
                blocks_assigned_to_center.append(block)
                population_assigned_to_center += area_df.loc[block]['population']
                assignment_ix += 1

            for block in blocks_assigned_to_center:
                unassigned_blocks.remove(block)

        centers.append(center_seed)
        block_seed = center_seed
        child_ix += 1
    return centers


def kmeans_seeds(state_df, n_distrs, n_random_seeds=0, perturbation_scale=None):

    weights = state_df.population.values + 1
    if perturbation_scale:
        weights = weight_perturbation(weights, perturbation_scale)
    if n_random_seeds:
        weights = rand_seed_reweight(weights, n_random_seeds)

    pts = state_df[['x', 'y']].values

    kmeans = KMeans(n_clusters=n_distrs, n_jobs=-1) \
        .fit(pts, sample_weight=weights).cluster_centers_

    dists = cdist(kmeans, pts)
    centers = [state_df.index[i].item()  # Convert to native int for jsonability
               for i in list(np.argmin(dists, axis=1))]

    return centers


def rand_seed_reweight(weights, n_seeds):
    n_seeds = int(n_seeds // 1 + (random.random() < n_seeds % 1))
    total_weight = weights.sum()
    for _ in range(n_seeds):
        rand_seed = random.randint(0, len(weights) - 1)
        weights[rand_seed] = total_weight
    return weights


def weight_perturbation(weights, scale):
    return weights * np.random.pareto(scale, len(weights))


def get_capacities(centers, child_sizes, area_df, config):
    n_children = len(child_sizes)
    total_seats = int(sum(child_sizes))

    center_locs = area_df.loc[centers][['x', 'y']].values
    locs = area_df[['x', 'y']].values
    pop = area_df['population'].values

    dist_mat = cdist(locs, center_locs)
    if config['capacity_weights'] == 'fractional':
        dist_mat **= 2
        weights = dist_mat / np.sum(dist_mat, axis=1)[:, None]
    elif config['capacity_weights'] == 'voronoi':
        assignment = np.argmin(dist_mat, axis=1)
        weights = np.zeros((len(locs), len(centers)))
        weights[np.arange(len(assignment)), assignment] = 1
    else:
        raise ValueError('Invalid capacity weight method')

    center_assignment_score = np.sum(weights * pop[:, None], axis=0)
    center_assignment_score /= center_assignment_score.sum()
    center_fractional_caps = center_assignment_score * total_seats

    if config['capacities'] == 'compute':
        cap_constraint = config.get('capacity_constraint', None)
        if cap_constraint:
            lb = max(1, math.floor(total_seats / (n_children * cap_constraint)))
            ub = min(total_seats, math.ceil((total_seats * cap_constraint) / n_children))
        else:
            lb = 1
            ub = total_seats
        center_caps = np.ones(n_children).astype(int) * lb
        while center_caps.sum() != total_seats:
            disparity = center_fractional_caps - center_caps
            at_capacity = center_caps >= ub
            disparity[at_capacity] = -total_seats  # enforce upperbound
            center_caps[np.argmax(disparity)] += 1

        return {center: capacity for center, capacity in zip(centers, center_caps)}

    elif config['capacities'] == 'match':
        center_order = center_assignment_score.argsort()
        capacities_order = child_sizes.argsort()

        return {centers[cen_ix]: child_sizes[cap_ix] for cen_ix, cap_ix
                in zip(center_order, capacities_order)}
    else:
        raise ValueError('Invalid capacity domain')


