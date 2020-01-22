import math
import numpy as np
import networkx as nx
from gerrypy.optimize.kmeans import *
from gerrypy.optimize.problems.splitter import make_splitter


def split(config, G, state_df, lengths):
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


def sample_node(config, G, state_df, lengths, split_info):
    """From a node in a the compatibility tree, sample k nodes"""
    if split_info['n_districts'] == 1:
        return [(split_info['split_area'], 1)]

    area = split_info['split_area']
    state_df = state_df.loc[area]
    area_lengths = {i: lengths[i] for i in area}

    samples = []
    for i in range(split_info['n_samples']):
        child_nodes = sample_random(config, G, state_df,
                                    area_lengths, split_info)
        if child_nodes:
            samples.append(child_nodes)
    if len(samples) == 0:
        raise ValueError('Unable to sample node')
    return [node for sample in samples for node in sample]


def sample_random(config, G, state_df, lengths, split_info):
    """Using a random seed, try k times to sample one split from a
    compatibility tree node."""
    for j in range(split_info['n_sample_tries']):
        centers = euclidean_kmeans_seeds({'n_districts': 2},
                                         state_df, random_seeds=1)
        tp_lengths = {i: {j: lengths[i].get(j, 9999)
                          for j in split_info['split_area']}
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
            return [(area, pop_bounds[center]['n_districts'])
                    for center, area in districting.items()]
    return []


def make_pop_bounds(split_info, state_df, centers):
    """Create a dictionary that records upper and lower bounds for
    population in addition to the number of districts the area contains."""
    c1, c2 = centers
    area = split_info['split_area']
    n_distrs = split_info['n_districts']
    if n_distrs % 2 == 0:
        distr_pop = state_df.population.sum() / 2
        ub = distr_pop * (1 + split_info['population_tolerance'])
        lb = distr_pop * (1 - split_info['population_tolerance'])
        pop_bounds = {
            c1: {'ub': ub, 'lb': lb, 'n_districts': n_distrs // 2},
            c2: {'ub': ub, 'lb': lb, 'n_districts': n_distrs // 2}
        }
    else:
        area_locs = state_df[['x', 'y']].values
        c1_loc = state_df.loc[c1][['x', 'y']].values
        c2_loc = state_df.loc[c2][['x', 'y']].values
        c1_max = np.max(np.linalg.norm(area_locs - c1_loc, axis=1))
        c2_max = np.max(np.linalg.norm(area_locs - c2_loc, axis=1))
        # Make the more centered tract be the bigger district
        bg_distr, sm_distr = (c1, c2) if c1_max < c2_max else (c2, c1)
        bg_distr_factor = (n_distrs // 2 + 1) / n_distrs
        bg_distr_pop = state_df.population.sum() * bg_distr_factor
        sm_distr_factor = (n_distrs // 2) / n_distrs
        sm_distr_pop = state_df.population.sum() * sm_distr_factor
        pop_bounds = {
            sm_distr: {
                'ub': sm_distr_pop * (1 + split_info['population_tolerance']),
                'lb': sm_distr_pop * (1 - split_info['population_tolerance']),
                'n_districts': n_distrs // 2
            },
            bg_distr: {
                'ub': bg_distr_pop * (1 + split_info['population_tolerance']),
                'lb': bg_distr_pop * (1 - split_info['population_tolerance']),
                'n_districts': n_distrs // 2 + 1
            }
        }
    return pop_bounds
