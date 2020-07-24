import os
import time
import json
import networkx as nx
from gerrypy.analyze.districts import *
from gerrypy.data.load import load_opt_data
from gerrypy.optimize.center_selection import *
from gerrypy.optimize.partition import make_partition_IP
from gerrypy.optimize.tree import SHPNode
import gerrypy.constants as consts


class ColumnGenerator:
    def __init__(self, config):
        state_abbrev = config['state']
        state_df, G, lengths, edge_dists = load_opt_data(state_abbrev)
        lengths /= 1000

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

        self.failed_regions = []
        self.failed_root_samples = 0
        self.n_infeasible_partitions = 0
        self.n_successful_partitions = 0

        self.event_list = []

    def generate(self):
        completed_root_samples = 0
        n_root_samples = self.config['n_root_samples']

        root = SHPNode(self.config['n_districts'],
                       list(self.state_df.index),
                       is_root=True)
        self.root = root
        self.internal_nodes.append(root)

        while completed_root_samples < n_root_samples:
            self.sample_queue = [root]
            sample_leaf_nodes = []
            sample_internal_nodes = []
            try:
                print('Root sample number', completed_root_samples)
                while len(self.sample_queue) > 0:
                    node = self.sample_queue.pop()
                    child_samples = self.sample_node(node)
                    if len(child_samples) == 0:
                        self.failed_regions.append(node.area)
                        if self.config['verbose']:
                            print(node.area)
                        raise RuntimeError('Unable to sample tree')
                    for child in child_samples:
                        if child.n_districts == 1:
                            sample_leaf_nodes.append(child)
                        else:
                            self.sample_queue.append(child)
                    if not node.is_root:
                        sample_internal_nodes.append(node)
                self.internal_nodes += sample_internal_nodes
                self.leaf_nodes += sample_leaf_nodes
                completed_root_samples += 1
            except RuntimeError:
                print('Root sample failed')
                self.root.children_ids = self.root.children_ids[:-1]
                self.root.partition_times = self.root.partition_times[:-1]
                self.failed_root_samples += 1

    def sample_node(self, node):
        """From a node in the compatibility tree, sample k children"""
        area_df = self.state_df.loc[node.area]
        samples = []
        n_trials = 0
        n_samples = 1 if node.is_root else self.config['n_samples']
        if not isinstance(n_samples, int):
            n_samples = int(n_samples // 1 + random.random() < n_samples % 1)
        while len(samples) < n_samples and n_trials < self.config['max_sample_tries']:
            partition_start_t = time.time()
            child_nodes = self.make_partition(area_df, node)
            partition_end_t = time.time()
            if child_nodes:
                self.n_successful_partitions += 1
                samples.append(child_nodes)
                node.children_ids.append([child.id for child in child_nodes])
                node.partition_times.append(partition_end_t - partition_start_t)
            else:
                self.n_infeasible_partitions += 1
                node.n_infeasible_samples += 1
            n_trials += 1

        return [node for sample in samples for node in sample]



    def make_partition(self, area_df, node):
        """Using a random seed, try w times to sample one split from a
        compatibility tree node."""
        children_sizes = node.sample_n_splits_and_child_sizes(self.config)

        # dict : {center_ix : child size}
        children_centers = self.select_centers(area_df, children_sizes)

        tp_lengths = {i: {j: self.lengths[i, j] for j in node.area}
                      for i in children_centers}

        pop_bounds = self.make_pop_bounds(children_centers)

        partition_IP, xs = make_partition_IP(tp_lengths,
                                         self.edge_dists,
                                         self.G,
                                         area_df.population.to_dict(),
                                         pop_bounds,
                                         1 + random.random())
        #partition_IP.Params.MIPGap = self.config['IP_gap_tol']
        partition_IP.update()
        partition_IP.optimize()
        try:
            districting = {i: [j for j in xs[i] if xs[i][j].X > .5]
                           for i in children_centers}
            feasible = all([nx.is_connected(nx.subgraph(self.G, distr)) for
                            distr in districting.values()])
        except AttributeError:
            feasible = False

        if self.config['event_logging']:
            if feasible:
                self.event_list.append({
                    'partition': districting,
                    'feasible': True,
                })
            else:
                self.event_list.append({
                    'area': node.area,
                    'centers': children_centers,
                    'feasible': False,
                })

        if self.config['verbose']:
            if feasible:
                print('successful sample')
            else:
                print('infeasible')

        if feasible:
            return [SHPNode(pop_bounds[center]['n_districts'], area)
                    for center, area in districting.items()]
        else:
            node.n_infeasible_samples += 1
            return []

    def select_centers(self, area_df, children_sizes):
        method = self.config['selection_method']
        if method == 'random_iterative':
            pop_capacity = self.config['ideal_pop'] * np.array(children_sizes)
            centers = iterative_random(self.config, area_df, pop_capacity, self.lengths)
        elif method == 'uncapacitated_kmeans':
            weight_perturbation_scale = self.config['perturbation_scale']
            n_random_seeds = self.config['n_random_seeds']
            centers = kmeans_seeds(area_df, len(children_sizes),
                                   n_random_seeds, weight_perturbation_scale)
        else:
            raise ValueError('center selection_method not valid')

        center_capacities = get_capacities(centers, children_sizes,
                                           area_df, self.config)

        return center_capacities

    def make_pop_bounds(self, children_centers):
        """Create a dictionary that records upper and lower bounds for
        population in addition to the number of districts the area contains."""
        pop_deviation = self.config['max_pop_variation']
        pop_bounds = {}
        # Make the bounds for an area considering # area districts and tree level
        for center, n_child_districts in children_centers.items():
            levels_to_leaf = max(math.ceil(math.log2(n_child_districts)), 1)
            distr_pop = self.config['ideal_pop'] * n_child_districts

            ub = distr_pop + pop_deviation / levels_to_leaf
            lb = distr_pop - pop_deviation / levels_to_leaf

            pop_bounds[center] = {
                'ub': ub,
                'lb': lb,
                'n_districts': n_child_districts
            }

        return pop_bounds

    def make_viz_list(self):
        n_districtings = 'nd' + str(number_of_districtings(self))
        n_leaves = 'nl' + str(len(self.leaf_nodes))
        n_interior = 'ni' + str(len(self.internal_nodes))
        width = 'w' + str(self.config['n_samples'])
        sample_tries = 'st' + str(self.config['n_sample_tries'])
        n_districts = 'ndist' + str(self.config['n_districts'])
        save_time = str(int(time.time()))
        save_name = '_'.join([self.state_abbrev, n_districtings, n_leaves,
                              n_interior, width, sample_tries, n_districts, save_time])

        save_path = os.path.join(consts.COLUMNS_PATH,
                                 self.state_abbrev,
                                 save_name)

        json.dump(self.event_list, open(save_path, 'w'))


if __name__ == '__main__':
    center_selection_config = {
        'selection_method': 'uncapacitated_kmeans',  # one of
        'perturbation_scale': 0,
        'n_random_seeds': 1,
        'capacities': 'compute',
        'capacity_weights': 'voronoi'
    }
    tree_config = {
        'max_sample_tries': 25,
        'n_samples': 3,
        'n_root_samples': 1,
        'max_n_splits': 5,
        'min_n_splits': 2,
        'max_split_population_difference': 1.5,
        'event_logging': False,
        'verbose': True,
    }
    gurobi_config = {
        'master_abs_gap': 1e-3,
        'master_max_time': 5,
        'IP_gap_tol': 1e-3,
        'IP_timeout': 10,
    }
    pdp_config = {
        'state': 'TX',
        'n_districts': 36,
        'population_tolerance': .01,
    }
    base_config = {**center_selection_config,
                   **tree_config,
                   **gurobi_config,
                   **pdp_config}
    cg = ColumnGenerator(base_config)
    cg.generate()