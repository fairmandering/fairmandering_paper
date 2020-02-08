import math
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from gerrypy.optimize.kmeans import *
from gerrypy.optimize.problems.splitter import make_splitter


def bfs_split(config, G, state_df, lengths):
    """Run hierarchical sampling """
    area = list(state_df.index)
    parent_nodes = [(area, config['n_districts'])]
    split_info = config['hconfig']
    n_layers = math.ceil(math.log(config['n_districts'], 2))
    for layer in range(n_layers):
        layer_children = []
        for node_area, n_districts in parent_nodes:
            print(layer, n_districts)
            split_info['n_districts'] = n_districts
            split_info['split_area'] = node_area
            split_info['cost_exponential'] = config['cost_exponential']
            pop_tol = config['population_tolerance'] * (layer + 1) / n_layers
            split_info['population_tolerance'] = pop_tol
            children = sample_node(config, G, state_df, lengths, split_info)
            layer_children.append(children)
        parent_nodes = [child for node in layer_children for child in node]
    return [district for district, _ in parent_nodes]


def non_binary_bfs_split(config, G, state_df, lengths, tree):
    sample_queue = []
    node_info = {
        'n_districts': tree.n_districts,
        'tree': tree,
        'area': list(state_df.index),
    }
    config['hconfig']['cost_exponential'] = config['cost_exponential']
    config['hconfig']['population_tolerance'] = config['population_tolerance']

    sample_queue.append(node_info)

    final_columns = []
    while len(sample_queue) > 0:
        node_info = sample_queue.pop()
        split_info = {**node_info, **config['hconfig']}  # Merge dicts
        child_samples = sample_node(config, G, state_df, lengths, split_info)
        if len(child_samples) == 0:
            raise ValueError('Unable to sample')
        for child in child_samples:
            if child['n_districts'] == 1:
                final_columns.append(child['area'])
            else:
                sample_queue.append(child)

    return final_columns


def sample_node(config, G, state_df, lengths, split_info):
    """From a node in a the compatibility tree, sample k nodes"""
    if split_info['n_districts'] == 1:
        return [(split_info['area'], 1)]

    area = split_info['area']
    state_df = state_df.loc[area]

    samples = []
    for i in range(split_info['n_samples']):
        child_nodes = sample_random(config, G, state_df,
                                    lengths, split_info)
        if child_nodes:
            samples.append(child_nodes)
        else:
            print('Unable to sample')
    if len(samples) == 0:
        raise ValueError('Unable to sample node')
    return [node for sample in samples for node in sample]


def sample_random(config, G, state_df, lengths, split_info):
    """Using a random seed, try k times to sample one split from a
    compatibility tree node."""
    for j in range(split_info['n_sample_tries']):
        n_centers = len(split_info['tree'].children)
        centers = euclidean_kmeans_seeds({'n_districts': n_centers},
                                         state_df, random_seeds=1)
        tp_lengths = {i: {j: lengths[i, j]
                          for j in split_info['area']}
                      for i in centers}

        pop_bounds = make_pop_bounds(split_info, state_df, centers)
        splitter, xs = make_splitter(split_info, tp_lengths,
                                     state_df.population.to_dict(),
                                     pop_bounds)
        splitter.update()
        splitter.optimize()
        districting = {i: [j for j in xs[i] if xs[i][j].X > .5]
                       for i in centers}
        connected = all([nx.is_connected(nx.subgraph(G, distr)) for
                         distr in districting.values()])
        if connected:
            return [{
                'area': area,
                'tree': pop_bounds[center]['child'],
                'n_districts': pop_bounds[center]['n_districts']
            } for center, area in districting.items()]
        else:
            print('Disconnected Sample')
    return []


def make_pop_bounds(split_info, state_df, centers):
    """Create a dictionary that records upper and lower bounds for
    population in addition to the number of districts the area contains."""
    # The number of districts this area contains
    n_districts = split_info['tree'].n_districts

    children = split_info['tree'].children

    area_pop = state_df.population.sum()
    bound_list = []
    # Make the bounds for an area considering # area districts and tree level
    for child in children:
        n_child_districts = child.n_districts
        levels_to_leaf = child.max_levels_to_leaf
        distr_pop = area_pop * n_child_districts / n_districts
        pop_tol = split_info['population_tolerance'] / (levels_to_leaf + 2)
        ub = distr_pop * (1 + pop_tol)
        lb = distr_pop * (1 - pop_tol)
        bound_list.append({
            'ub': ub,
            'lb': lb,
            'n_districts': n_child_districts,
            'child': child
        })

    # Make most centralized center have most number of districts
    # Centers closer to area borders have less districts
    bound_list.sort(key=lambda x: x['n_districts'])
    area_locs = state_df[['x', 'y', 'z']].values
    center_locs = state_df.loc[centers, ['x', 'y', 'z']].values
    center_max_dists = np.max(cdist(area_locs, center_locs), axis=0)
    center_max_dists = [(center, dist) for center, dist
                        in zip(centers, center_max_dists)]
    center_max_dists.sort(key=lambda x: x[1], reverse=True)

    pop_bounds = {center: bounds for (center, _), bounds
                  in zip(center_max_dists, bound_list)}
    return pop_bounds
