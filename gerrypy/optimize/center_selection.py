import math
import random
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from gerrypy.utils.spatial_utils import *


def iterative_random(cs_config, area_df, capacities, pdists):
    unassigned_blocks = list(area_df.index)

    if cs_config['center_assignment_order'] == 'random':
        np.random.shuffle(capacities)

    elif cs_config['center_assignment_order'] == 'ascending':
        capacities = np.sort(capacities)

    elif cs_config['center_assignment_order'] == 'descending':
        capacities = np.sort(capacities)[::-1]

    else:
        raise ValueError('Not a valid center_assignment_order')

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

def random_centers(cs_config, area_df, capacities, pdists):
    unassigned_blocks = list(area_df.index)

    if cs_config['center_assignment_order'] == 'random':
        np.random.shuffle(capacities)

    elif cs_config['center_assignment_order'] == 'ascending':
        capacities = np.sort(capacities)

    elif cs_config['center_assignment_order'] == 'descending':
        capacities = np.sort(capacities)[::-1]

    else:
        raise ValueError('Not a valid center_assignment_order')

    center_assignment = {}
    centers = []
    child_ix = 0
    while child_ix < len(capacities):
        block_seed = random.choice(unassigned_blocks)
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
        else:
            blocks_assigned_to_center = unassigned_blocks.copy()

        center_assignment[center_seed] = {
            'child_ix': child_ix,
            'size': capacities[child_ix],
            'assigned': blocks_assigned_to_center
        }

        if cs_config['seed_to_center_method'] == 'identity':
            centers.append(center_seed)

        elif cs_config['seed_to_center_method'] == 'uniform_random':
            centers.append(random.choice(center_assignment))

        elif cs_config['seed_to_center_method'] == 'distance_weighted_random':
            center_seed_sq_dist = 1 + pdists[center_seed, blocks_assigned_to_center] ** 2
            center_p = center_seed_sq_dist / np.sum(center_seed_sq_dist)
            centers.append(np.random.choice(blocks_assigned_to_center, p=center_p))

        elif cs_config['seed_to_center_method'] == 'centroid':
            assigned_locs = area_df[['x', 'y']].loc[center_assignment].values
            centroid = np.mean(assigned_locs, axis=0)
            centroid_dists = assigned_locs.subtract(centroid)
            center = center_assignment[np.argmin(np.linalg.norm(centroid_dists))]
            centers.append(center)

        else:
            raise ValueError('Not a valid seed_to_center_method')

        child_ix += 1
    return centers, center_assignment


def kmeans_initialization(capacities, area_df, n_random, init='++ prob'):
    k = len(capacities)
    n_blocks = len(area_df)
    locations = area_df[['x', 'y']].values

    if init == 'random':
        centers = np.random.choice(n_blocks, k, replace=False)
    else:
        # init k centers using kmeans++ initialization variant
        seed = np.random.choice(n_blocks)
        centers = [seed]
        first_round = True
        while len(centers) < k:
            # squared distance to closest center
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


def capacitated_kmeans(capacities, area_df, n_random=0, init='cap_random'):
    k = len(capacities)
    locations = area_df[['x', 'y']].values

    centers = kmeans_initialization(capacities, area_df, n_random, init)
    # initialize other data structures
    prev_centers = set()

    random_mask = np.ones(k)
    random_mask[np.random.choice(k, n_random, replace=False)] = 0

    while set(centers) != prev_centers:
        # calculate distances
        dists = cdist(locations, locations[centers])
        search_order = np.argsort(dists.min(axis=1))

        fill_factor = np.ones(k)

        center_assignment = [[] for _ in range(k)]

        # iterate over precincts and update priorities
        for precinct_ix in search_order:
            priority = dists[precinct_ix] * fill_factor ** 2
            selected_center_ix = np.argmin(priority)
            center_assignment[selected_center_ix].append(precinct_ix)

            fill_adjustment = area_df.iloc[precinct_ix]['population'] / capacities[selected_center_ix]
            fill_factor[selected_center_ix] += fill_adjustment

        # calulate new centers
        centroid_positions = np.array([
            np.average(locations[precinct_ids], axis=0,
                       weights=area_df.iloc[precinct_ids].population.values)
            for precinct_ids in center_assignment
        ])

        centroid_dists = cdist(locations, centroid_positions)

        prev_centers = set(centers)
        centers = list(np.argmin(centroid_dists, axis=0))

    return list(area_df.iloc[centers].index)


def kmeans_seeds(state_df, n_distrs, n_random_seeds=0, perturbation_scale=None):

    weights = state_df.population.values + 1
    if perturbation_scale:
        weights = weight_perturbation(weights, perturbation_scale)
    if n_random_seeds:
        weights = rand_seed_reweight(weights, n_random_seeds)

    pts = state_df[['x', 'y']]

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


def get_capacities(centers, child_sizes, area_df, kwargs):
    n_children = len(child_sizes)
    total_seats = int(sum(child_sizes))

    center_locs = area_df.loc[centers][['x', 'y']].values
    locs = area_df[['x', 'y']].values
    pop = area_df['population'].values

    dist_mat = cdist(locs, center_locs)
    if kwargs['weights'] == 'fractional':
        dist_mat **= kwargs['dist_penalty']
        weights = dist_mat / np.sum(dist_mat, axis=1)[:, None]
    elif kwargs['weights'] == 'voronoi':
        assignment = np.argmin(dist_mat, axis=1)
        weights = np.zeros((len(locs), len(centers)))
        weights[np.arange(len(assignment)), assignment] = 1
    else:
        raise ValueError('Invalid capacity weight method')

    center_assignment_score = np.sum(weights * pop[:, None], axis=0)
    center_assignment_score /= center_assignment_score.sum()
    center_fractional_caps = center_assignment_score * total_seats

    if kwargs['capacities'] == 'compute':
        cap_constraint = kwargs.get('capacity_constraint', None)
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

    elif kwargs['capacities'] == 'match':
        center_order = center_assignment_score.argsort()
        capacities_order = child_sizes.argsort()

        return {centers[cen_ix]: child_sizes[cap_ix] for cen_ix, cap_ix
                in zip(center_order, capacities_order)}
    else:
        raise ValueError('Invalid capacity domain')


def assign_children_to_centers(centers, child_sizes, area_df):
    center_locs = area_df.loc[centers][['x', 'y']].values
    locs = area_df[['x', 'y']].values
    pop = area_df['population'].values

    dist_mat = cdist(locs, center_locs)
    dist_mat /= np.sum(dist_mat, axis=1)[:, None]

    center_assignment_score = np.sum(dist_mat * pop[:, None], axis=0)

    center_order = center_assignment_score.argsort()
    capacities_order = child_sizes.argsort()

    return {centers[cen_ix]: child_sizes[cap_ix] for cen_ix, cap_ix
            in zip(center_order, capacities_order)}


def compute_fractional_capacities(centers, child_sizes, area_df):
    n_children = len(child_sizes)
    total_size = int(sum(child_sizes))

    center_locs = area_df.loc[centers][['x', 'y']].values
    locs = area_df[['x', 'y']].values
    pop = area_df['population'].values

    dist_mat = cdist(locs, center_locs)
    dist_mat /= np.sum(dist_mat, axis=1)[:, None]

    center_assignment_score = np.sum(dist_mat * pop[:, None], axis=0)
    center_assignment_score /= center_assignment_score.sum()
    center_fractional_caps = center_assignment_score * total_size

    center_caps = np.ones(n_children).astype(int)
    while center_caps.sum() != total_size:
        disparity = center_fractional_caps - center_caps
        center_caps[np.argmax(disparity)] += 1
    print(center_caps)

    return {center: capacity for center, capacity in zip(centers, center_caps)}


def compute_optimal_capacities(centers, child_sizes, area_df):
    n_children = len(child_sizes)
    total_size = int(sum(child_sizes))

    center_locs = area_df.loc[centers][['x', 'y']].values
    locs = area_df[['x', 'y']].values
    pop = area_df['population'].values

    dist_mat = cdist(locs, center_locs)
    assignment = np.argmin(dist_mat, axis=1)

    assignment_ix = np.zeros((len(locs), len(centers)))
    assignment_ix[np.arange(len(assignment)), assignment] = 1

    center_assignment_score = np.sum(assignment_ix * pop[:, None], axis=0)
    center_assignment_score /= center_assignment_score.sum()
    center_fractional_caps = center_assignment_score * total_size

    center_caps = np.ones(n_children).astype(int)
    while center_caps.sum() != total_size:
        disparity = center_fractional_caps - center_caps
        center_caps[np.argmax(disparity)] += 1
    print(center_caps)

    return {center: capacity for center, capacity in zip(centers, center_caps)}

