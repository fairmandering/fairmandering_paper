import os
import time
import json
import itertools
from gerrypy.analyze.districts import *
from gerrypy.optimize.main import load_real_data
from gerrypy.optimize.center_selection import *
from gerrypy.optimize.partition import make_partition_IP
from gerrypy.optimize.tree import SHPNode
import gerrypy.constants as consts


class ColumnGenerator:
    def __init__(self, config, state_abbrev):
        # TODO: Fix data loading
        state_fips = consts.ABBREV_DICT[state_abbrev][consts.FIPS_IX]
        data_path = os.path.join(consts.OPT_DATA_PATH, str(state_fips))
        state_df, G, lengths, edge_dists = load_real_data(data_path)

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
                self.failed_root_samples += 1

    def sample_node(self, node):
        """From a node in the compatibility tree, sample k children"""
        area_df = self.state_df.loc[node.area]
        samples = []
        n_trials = 0
        n_samples = 1 if node.is_root else self.config['n_samples']
        while len(samples) < n_samples and n_trials < self.config['max_sample_tries']:
            child_nodes = self.partition(area_df, node)
            if child_nodes:
                self.n_successful_partitions += 1
                samples.append(child_nodes)
                node.children_ids.append([child.id for child in child_nodes])
            else:
                self.n_infeasible_partitions += 1
            n_trials += 1

        return [node for sample in samples for node in sample]

    def sample_node_old(self, node):
        """From a node in the compatibility tree, sample k children"""
        area_df = self.state_df.loc[node.area]

        samples = []
        n_samples = range(1) if node.is_root else range(self.config['n_samples'])
        for _ in n_samples:
            child_nodes = self.partition(area_df, node)
            if child_nodes:
                samples.append(child_nodes)
                node.children_ids.append([child.id for child in child_nodes])
            else:
                node.n_sample_failures += 1

        return [node for sample in samples for node in sample]

    def make_partition_IP(self, area_df, node):
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
        cs_config = self.config['center_selection_config']
        method = cs_config['selection_method']
        if method == 'random_iterative':
            pop_capacity = self.config['ideal_pop'] * np.array(children_sizes)
            centers = iterative_random(cs_config, area_df, pop_capacity, self.lengths)
        elif method == 'uncapacitated_kmeans':
            weight_perturbation_scale = cs_config['perturbation_scale']
            n_random_seeds = cs_config['n_random_seeds']
            centers = kmeans_seeds(area_df, len(children_sizes),
                                   n_random_seeds, weight_perturbation_scale)
        elif method == 'capacitated_kmeans':
            raise NotImplementedError('TODO: debug capacitated kmeans')
        else:
            raise ValueError('center selection_method not valid')

        cap_kwargs = cs_config['capacity_kwargs']
        center_capacities = get_capacities(centers, children_sizes,
                                           area_df, cap_kwargs)

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

    def district_metrics(self):
        def average_entropy(M):
            return (- M * np.ma.log(M).filled(0) - (1 - M) *
                    np.ma.log(1 - M).filled(0)).sum() / (M.shape[0] * M.shape[1])

        def svd_entropy(sigma):
            sigma_hat = sigma / sigma.sum()
            entropy = - (sigma_hat * (np.ma.log(sigma_hat).filled(0) / np.log(math.e))).sum()
            return entropy / (math.log(len(sigma)) / math.log(math.e))

        p_infeasible = self.n_infeasible_partitions / \
                       (self.n_infeasible_partitions + self.n_successful_partitions)
        n_interior_nodes = len(self.internal_nodes)
        districts = [d.area for d in self.leaf_nodes]
        duplicates = len(districts) - len(set([frozenset(d) for d in districts]))

        block_district_matrix = np.zeros((len(self.state_df), len(districts)))
        for ix, d in enumerate(districts):
            block_district_matrix[d, ix] = 1
        ubdm = np.unique(block_district_matrix, axis=1)

        max_rank = min(ubdm.shape)
        U, Sigma, Vt = np.linalg.svd(ubdm)

        Dsim = 1 - pdist(ubdm.T, metric='jaccard')

        precinct_coocc = ubdm @ ubdm.T
        precinct_conditional_p = precinct_coocc / ubdm.sum(axis=1)

        L = (np.diag(precinct_coocc.sum(axis=1)) - precinct_coocc)
        D_inv = np.diag(precinct_coocc.sum(axis=1) ** -.5)
        e = np.linalg.eigvals(D_inv @ L @ D_inv)

        conditional_entropy = average_entropy(precinct_conditional_p)

        dispersion = dispersion_compactness(districts, self.state_df)
        roeck = roeck_compactness(districts, self.state_df, self.lengths)

        metrics = {
            'n_root_failures': self.failed_root_samples,
            'p_infeasible': p_infeasible,
            'n_interior_nodes': n_interior_nodes,
            'n_districts': len(districts),
            'p_duplicates': duplicates / len(districts),
            'conditional_entropy': conditional_entropy,
            'average_district_sim': self.config['n_districts'] * np.average(Dsim),
            'svd_entropy': svd_entropy(Sigma),
            '50p_approx_rank': sum(np.cumsum(Sigma / Sigma.sum()) < .5) / max_rank,
            '95p_approx_rank': sum(np.cumsum(Sigma / Sigma.sum()) < .95) / max_rank,
            '99p_approx_rank': sum(np.cumsum(Sigma / Sigma.sum()) < .99) / max_rank,
            'lambda_2': e[1],
            'lambda_k': e[self.config['n_districts']],
            'dispersion': np.array(dispersion).mean(),
            'roeck': np.array(roeck).mean()
        }

        return metrics, Sigma, block_district_matrix
