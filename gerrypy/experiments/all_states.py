import os
import pickle
import time
import copy
from gerrypy import constants
from gerrypy.analyze.districts import number_of_districtings
from gerrypy.optimize.generate import ColumnGenerator


class StateRunner:
    def __init__(self, name, base_config):
        self.base_config = base_config
        self.name = name

    def k_to_w(self, k):
        w_root = int(round((2000 / k) ** 1.2))
        w_internal = int(round((200 / k) ** .7))
        return w_root, w_internal

    def run(self, test=False, test_sample_widths=(1, 1)):
        experiment_dir_name = self.name + str(int(time.time()))
        os.makedirs(experiment_dir_name)
        for state, seats_dict in constants.seats.items():
            if seats_dict['house'] > 1:
                if test:
                    w_root, w_internal = test_sample_widths
                else:
                    w_root, w_internal = self.k_to_w(seats_dict['house'])
                trial_config = copy.deepcopy(self.base_config)
                trial_config['state'] = state
                trial_config['n_districts'] = seats_dict['house']
                trial_config['n_samples'] = w_internal
                trial_config['n_root_samples'] = w_root
                cg = ColumnGenerator(trial_config)
                start_t = time.time()
                cg.generate()
                end_t = time.time()
                n_plans = number_of_districtings(cg.leaf_nodes, cg.internal_nodes)
                result_dict = {
                    'generation_time': end_t - start_t,
                    'leaf_nodes': cg.leaf_nodes,
                    'internal_nodes': cg.internal_nodes,
                    'trial_config': trial_config,
                    'n_plans': n_plans
                }
                file_name = '%s_%d.p' % (state, n_plans)
                file_path = os.path.join(experiment_dir_name, file_name)
                pickle.dump(result_dict, open(file_path, 'wb'))


if __name__ == '__main__':
    center_selection_config = {
        'selection_method': 'uncapacitated_kmeans',  # one of
        'perturbation_scale': 1,
        'n_random_seeds': 1,
        'capacities': 'match',
        'capacity_weights': 'voronoi',
        'use_subgraph': True
    }
    tree_config = {
        'max_sample_tries': 30,
        'n_samples': 3,
        'n_root_samples': 1,
        'max_n_splits': 5,
        'min_n_splits': 2,
        'max_split_population_difference': 1.5,
        'event_logging': False,
        'verbose': False,
    }
    gurobi_config = {
        'master_abs_gap': 1e-3,
        'master_max_time': 5,
        'IP_gap_tol': 1e-3,
        'IP_timeout': 10,
    }
    pdp_config = {
        'state': 'NC',
        'n_districts': 13,
        'population_tolerance': .01,
    }
    base_config = {**center_selection_config,
                   **tree_config,
                   **gurobi_config,
                   **pdp_config}
    experiment = StateRunner('aaai_columns', base_config)
    experiment.run(test=True, test_sample_widths=(2, 2))
