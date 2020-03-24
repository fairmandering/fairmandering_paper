import os
import math
import time
import json
import itertools
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from gerrypy.optimize.main import load_real_data
from gerrypy.optimize.kmeans import *
from gerrypy.optimize.problems.splitter import make_splitter
from gerrypy.optimize.problems.connected_splitter import make_connected_splitter
from gerrypy.optimize.tree import SHCNode
import gerrypy.constants as consts


class ColumnGenerator:
    def __init__(self, config, state_abbrev):
        state_fips = consts.ABBREV_DICT[state_abbrev][consts.FIPS_IX]
        data_path = os.path.join(consts.OPT_DATA_PATH, str(state_fips))
        state_df, G, _, lengths, edge_dists = load_real_data(data_path)

        self.state_fips = state_fips
        self.state_abbrev = state_abbrev

        ideal_pop = state_df.population.values.sum() / config['n_districts']
        max_pop_variation = ideal_pop * config['population_tolerance']

        config['max_pop_variation'] = max_pop_variation
        config['ideal_pop'] = ideal_pop

        self.config = config
        self.G = G
        self.state_df = state_df
        self.lengths = lengths
        self.edge_dists = edge_dists

        self.sample_queue = []
        self.internal_nodes = []
        self.leaf_nodes = []
        self.root = None

        self.event_list = []

    def generate(self):
        root = SHCNode(self.config['n_districts'], list(self.state_df.index))

        self.root = root
        self.sample_queue.append(root)

        while len(self.sample_queue) > 0:
            node = self.sample_queue.pop()
            child_samples = self.sample_node(node)
            if len(child_samples) == 0:
                raise RuntimeError('Unable to sample tree')
            for child in child_samples:
                if child.n_districts == 1:
                    self.leaf_nodes.append(child)
                else:
                    self.sample_queue.append(child)
            self.internal_nodes.append(node)

    def sample_node(self, node):
        """From a node in a the compatibility tree, sample k children"""
        area_df = self.state_df.loc[node.area]

        samples = []
        for i in range(self.config['n_samples']):
            child_nodes = self.partition(area_df, node)
            if child_nodes:
                samples.append(child_nodes)
                node.children_ids.append([child.id for child in child_nodes])
            else:
                node.n_sample_failures += 1

        return [node for sample in samples for node in sample]

    def partition(self, area_df, node):
        """Using a random seed, try k times to sample one split from a
        compatibility tree node."""
        for j in range(self.config['max_sample_tries']):
            children = node.sample_n_children(self.config)
            n_centers = len(children)
            centers = euclidean_kmeans_seeds({'n_districts': n_centers},
                                             area_df, random_seeds=1)

            tp_lengths = {i: {j: self.lengths[i, j] for j in node.area}
                          for i in centers}

            pop_bounds = self.make_pop_bounds(node, area_df, centers, children)

            if not self.config['enforce_connectivity']:
                splitter, xs = make_splitter(tp_lengths,
                                             area_df.population.to_dict(),
                                             pop_bounds,
                                             1 + random.random())

            else:
                splitter, xs = make_connected_splitter(tp_lengths,
                                                       self.edge_dists,
                                                       self.G,
                                                       area_df.population.to_dict(),
                                                       pop_bounds,
                                                       1 + random.random())
            splitter.update()
            splitter.optimize()
            try:
                districting = {i: [j for j in xs[i] if xs[i][j].X > .5]
                               for i in centers}
                connected = all([nx.is_connected(nx.subgraph(self.G, distr)) for
                                 distr in districting.values()])
            except AttributeError:
                connected = False


            if self.config['event_logging']:
                if connected:
                    self.event_list.append({
                        'partition': districting,
                        'feasible': True,
                    })
                else:
                    self.event_list.append({
                        'area': node.area,
                        'centers': centers,
                        'feasible': False,
                    })

            if self.config['verbose']:
                if connected:
                    print('successful sample')
                else:
                    if self.config['enforce_connectivity']:
                        print('infeasible')
                    else:
                        print('disconnected')

            if connected:
                return [SHCNode(pop_bounds[center]['n_districts'], area)
                        for center, area in districting.items()]
            else:
                node.n_disconnected_samples += 1
        return []

    def make_pop_bounds(self, node, area_df, centers, children):
        """Create a dictionary that records upper and lower bounds for
        population in addition to the number of districts the area contains."""
        # The number of districts this area contains
        pop_deviation = self.config['max_pop_variation']

        bound_list = []
        # Make the bounds for an area considering # area districts and tree level
        for n_child_districts in children:
            levels_to_leaf = math.ceil(math.log2(n_child_districts))
            levels_to_leaf = levels_to_leaf if levels_to_leaf >= 1 else 1
            distr_pop = self.config['ideal_pop'] * n_child_districts

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
        area_locs = area_df[['x', 'y']].values
        center_locs = area_df.loc[centers, ['x', 'y']].values
        center_max_dists = np.max(cdist(area_locs, center_locs), axis=0)
        center_max_dists = [(center, dist) for center, dist
                            in zip(centers, center_max_dists)]
        center_max_dists.sort(key=lambda x: x[1], reverse=True)

        pop_bounds = {center: bounds for (center, _), bounds
                      in zip(center_max_dists, bound_list)}
        return pop_bounds

    def enumerate_partitions(self):
        def feasible_partitions(node, node_dict):
            if not node.children_ids:
                return [[node.id]]

            partitions = []
            for disjoint_sibling_set in node.children_ids:
                sibling_partitions = []
                for child in disjoint_sibling_set:
                    sibling_partitions.append(feasible_partitions(node_dict[child],
                                                                  node_dict))
                combinations = [list(itertools.chain.from_iterable(combo))
                                for combo in itertools.product(*sibling_partitions)]
                partitions.append(combinations)

            return list(itertools.chain.from_iterable(partitions))

        node_dict = {n.id: n for n in self.internal_nodes + self.leaf_nodes}
        return feasible_partitions(self.root, node_dict)

    def number_of_districtings(self):
        nodes = self.leaf_nodes + self.internal_nodes
        id_to_node = {node.id: node for node in nodes}

        def recursive_compute(current_node, all_nodes):
            if not current_node.children_ids:
                return 1

            total_districtings = 0
            for sample in current_node.children_ids:
                sample_districtings = 1
                for child_id in sample:
                    child_node = id_to_node[child_id]
                    sample_districtings *= recursive_compute(child_node, all_nodes)

                total_districtings += sample_districtings
            return total_districtings

        return recursive_compute(self.root, nodes)

    def make_viz_list(self):
        n_districtings = 'nd' + str(self.number_of_districtings())
        n_leaves = 'nl' + str(len(self.leaf_nodes))
        n_interior = 'ni' + str(len(self.internal_nodes))
        width = 'w' + str(self.config['n_samples'])
        sample_tries = 'st' + str(self.config['n_sample_tries'])
        n_districts = 'ndist' + str(self.config['n_districts'])
        connectivity = 'fconn' if self.config['enforce_connectivity'] else 'uncon'
        save_time = str(int(time.time()))
        save_name = '_'.join([self.state_abbrev, n_districtings, n_leaves,
                              n_interior, width, sample_tries, n_districts,
                              connectivity, save_time])

        save_path = os.path.join(consts.COLUMNS_PATH,
                                 self.state_abbrev,
                                 save_name)

        json.dump(self.event_list, open(save_path, 'w'))
