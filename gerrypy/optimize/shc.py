import math
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from gerrypy.optimize.kmeans import *
from gerrypy.optimize.problems.splitter import make_splitter
from gerrypy.optimize.tree import SampleNode


def shc(config, G, state_df, lengths):
    sample_queue = []
    internal_nodes = []
    leaf_nodes = []

    config['hconfig']['cost_exponential'] = config['cost_exponential']
    config['hconfig']['population_tolerance'] = config['population_tolerance']

    root = SampleNode(config['hconfig'], config['n_districts'],
                      list(state_df.index))

    sample_queue.append(root)

    while len(sample_queue) > 0:
        node = sample_queue.pop()
        child_samples = sample_node(G, state_df, lengths, node)
        if len(child_samples) == 0:
            raise RuntimeError('Unable to sample tree')
        for child in child_samples:
            if child.n_districts == 1:
                leaf_nodes.append(child)
            else:
                sample_queue.append(child)
        internal_nodes.append(node)

    return leaf_nodes, internal_nodes


def sample_node(G, state_df, lengths, node):
    """From a node in a the compatibility tree, sample k children"""
    state_df = state_df.loc[node.area]

    samples = []
    for i in range(node.hconfig['n_samples']):
        child_nodes = sample_random(G, state_df, lengths, node)
        if child_nodes:
            samples.append(child_nodes)
            node.children_ids.append([child.id for child in child_nodes])
        else:
            node.n_sample_failures += 1

    return [node for sample in samples for node in sample]


def sample_random(G, state_df, lengths, node):
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
        splitter, xs = make_splitter(tp_lengths,
                                     state_df.population.to_dict(),
                                     pop_bounds, 1 + random.random())
        splitter.update()
        splitter.optimize()
        districting = {i: [j for j in xs[i] if xs[i][j].X > .5]
                       for i in centers}
        connected = all([nx.is_connected(nx.subgraph(G, distr)) for
                         distr in districting.values()])
        if connected:
            return [SampleNode(node.hconfig,
                               pop_bounds[center]['n_districts'],
                               area)
                    for center, area in districting.items()]

        else:
            node.n_disconnected_samples += 1
    return []


def make_pop_bounds(node, state_df, centers, children):
    """Create a dictionary that records upper and lower bounds for
    population in addition to the number of districts the area contains."""
    # The number of districts this area contains
    n_districts = node.n_districts

    area_pop = state_df.population.sum()
    pop_deviation = area_pop / n_districts * node.hconfig['population_tolerance']

    bound_list = []
    # Make the bounds for an area considering # area districts and tree level
    for n_child_districts in children:
        # levels_to_leaf = math.ceil(math.log2(n_child_districts))
        distr_pop = area_pop * n_child_districts / n_districts

        ub = distr_pop + pop_deviation
        lb = distr_pop - pop_deviation

        bound_list.append({
            'ub': ub,
            'lb': lb,
            'n_districts': n_child_districts
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
