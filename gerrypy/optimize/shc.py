import math
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from gerrypy.optimize.kmeans import *
from gerrypy.optimize.problems.splitter import make_splitter
from gerrypy.optimize.problems.connected_splitter import make_connected_splitter

from gerrypy.optimize.tree import SHCNode


def shc(config, G, state_df, lengths, edge_dists):
    sample_queue = []
    internal_nodes = []
    leaf_nodes = []

    ideal_pop = state_df.population.values.sum() / config['n_districts']
    max_pop_variation = ideal_pop * config['population_tolerance']

    config['hconfig']['max_pop_variation'] = max_pop_variation
    config['hconfig']['ideal_pop'] = ideal_pop
    enforce_connectivity = config['enforce_connectivity']

    root = SHCNode(config['hconfig'], config['n_districts'],
                   list(state_df.index))

    sample_queue.append(root)

    while len(sample_queue) > 0:
        node = sample_queue.pop()
        child_samples = sample_node(G, state_df, lengths, edge_dists, node, enforce_connectivity)
        if len(child_samples) == 0:
            raise RuntimeError('Unable to sample tree')
        for child in child_samples:
            if child.n_districts == 1:
                leaf_nodes.append(child)
            else:
                sample_queue.append(child)
        internal_nodes.append(node)

    return leaf_nodes, internal_nodes


def sample_node(G, state_df, lengths, edge_dists, node, enforce_connectivity):
    """From a node in a the compatibility tree, sample k children"""
    state_df = state_df.loc[node.area]

    samples = []
    for i in range(node.hconfig['n_samples']):
        child_nodes = sample_random(G, state_df, lengths, edge_dists, node, enforce_connectivity)
        if child_nodes:
            samples.append(child_nodes)
            node.children_ids.append([child.id for child in child_nodes])
        else:
            node.n_sample_failures += 1

    return [node for sample in samples for node in sample]


def sample_random(G, state_df, lengths, edge_dists, node, enforce_connectivity):
    """Using a random seed, try k times to sample one split from a
    compatibility tree node."""
    for j in range(node.hconfig['n_sample_tries']):
        children = node.sample_n_children()
        n_centers = len(children)
        centers = euclidean_kmeans_seeds({'n_districts': n_centers},
                                         state_df, random_seeds=1)
        tp_lengths = {i: {j: lengths[i, j] for j in node.area}
                      for i in centers}

        pop_bounds = make_pop_bounds(node, state_df, centers, children)

        if not enforce_connectivity:
            splitter, xs = make_splitter(tp_lengths,
                                         state_df.population.to_dict(),
                                         pop_bounds, 1 + random.random())

        else:
            splitter, xs = make_connected_splitter(tp_lengths, edge_dists, G,
                                         state_df.population.to_dict(),
                                         pop_bounds, 1 + random.random())
        splitter.update()
        splitter.optimize()
        try:
            districting = {i: [j for j in xs[i] if xs[i][j].X > .5]
                           for i in centers}
            connected = all([nx.is_connected(nx.subgraph(G, distr)) for
                             distr in districting.values()])
        except AttributeError:
            connected = False

        if connected:
            print('successful sample')
            return [SHCNode(node.hconfig,
                            pop_bounds[center]['n_districts'],
                            area)
                    for center, area in districting.items()]

        else:
            if enforce_connectivity:
                print('infeasible')
            else:
                print('disconnected')
            node.n_disconnected_samples += 1
    return []


def make_pop_bounds(node, state_df, centers, children):
    """Create a dictionary that records upper and lower bounds for
    population in addition to the number of districts the area contains."""
    # The number of districts this area contains
    n_districts = node.n_districts

    area_pop = state_df.population.sum()
    pop_deviation = node.hconfig['max_pop_variation']

    bound_list = []
    # Make the bounds for an area considering # area districts and tree level
    for n_child_districts in children:
        levels_to_leaf = math.ceil(math.log2(n_child_districts))
        levels_to_leaf = levels_to_leaf if levels_to_leaf >= 1 else 1
        distr_pop = node.hconfig['ideal_pop'] * n_child_districts

        ub = distr_pop + pop_deviation / levels_to_leaf
        lb = distr_pop - pop_deviation / levels_to_leaf

        bound_list.append({
            'ub': ub,
            'lb': lb,
            'n_districts': n_child_districts
        })

    # Make most centralized center have most number of districts
    # Centers closer to area borders have less districts
    bound_list.sort(key=lambda x: x['n_districts'])
    area_locs = state_df[['x', 'y']].values
    center_locs = state_df.loc[centers, ['x', 'y']].values
    center_max_dists = np.max(cdist(area_locs, center_locs), axis=0)
    center_max_dists = [(center, dist) for center, dist
                        in zip(centers, center_max_dists)]
    center_max_dists.sort(key=lambda x: x[1], reverse=True)

    pop_bounds = {center: bounds for (center, _), bounds
                  in zip(center_max_dists, bound_list)}
    return pop_bounds
