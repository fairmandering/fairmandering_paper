from gerrypy.optimize.generate import ColumnGenerator
from gerrypy.analyze.districts import *
from gerrypy.utils.dir_processing import district_df_of_tree_dir
from gerrypy import constants
from copy import deepcopy
import time
import os
import numpy as np
import json


class Experiment:
    def __init__(self, base_config, experiment_config):
        self.base_config = base_config
        self.experiment_config = experiment_config

    def run(self):
        name = self.experiment_config['name']
        experiment_dir = '%s_results_%s' % (name, str(int(time.time())))
        save_dir = os.path.join(constants.RESULTS_PATH, experiment_dir)
        os.mkdir(save_dir)
        for state in self.experiment_config['states']:
            print('############## Starting %s trials ##############' % state)
            for trial_values in self.experiment_config['trial_parameters']:
                trial_config = deepcopy(self.base_config)
                for (k, v) in trial_values:
                    if len(k) == 2:
                        trial_config[k[0]][k[1]] = v
                    else:
                        trial_config[k] = v
                # trial_config['n_districts'] = constants.seats[state]['house']
                trial_config['state'] = state

                print('Starting trial', trial_config)
                cg = ColumnGenerator(trial_config)
                generation_start_t = time.time()
                cg.generate()
                generation_t = time.time() - generation_start_t
                analysis_start_t = time.time()
                metrics = generation_metrics(cg, low_memory=self.experiment_config['low_memory'])
                analysis_t = time.time() - analysis_start_t

                trial_results = {
                    'generation_time': generation_t,
                    'analysis_time': analysis_t,
                    'leaf_nodes': cg.leaf_nodes,
                    'internal_nodes': cg.internal_nodes,
                    'trial_config': trial_config,
                    'trial_values': trial_values,
                    'metrics': metrics,
                    'n_plans': number_of_districtings(cg.leaf_nodes, cg.internal_nodes)
                }

                def process(val):
                    if isinstance(val, dict):
                        return ''.join([c for c in str(val) if c.isalnum()])
                    else:
                        return str(val)

                config_str = '_'.join([process(v) for k, v in trial_values])
                save_name = '_'.join([state, config_str, str(int(time.time()))]) + '.npy'
                # json.dump(trial_results, open(os.path.join(save_dir, save_name), 'w'))
                np.save(os.path.join(save_dir, save_name), trial_results)
        if self.experiment_config['create_district_df']:
            district_df_of_tree_dir(save_dir)



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
        'n_root_samples': 150,
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
    experiment_config = {
        'name': 'PNAS_NC_k',
        'states': ['NC'],
        'low_memory': True,
        'create_district_df': True,
        'trial_parameters': [
            [('n_districts', 5), ('population_tolerance', .0025), ('n_samples', 5), ('n_root_samples', 100)],
            [('n_districts', 10), ('population_tolerance', .005), ('n_samples', 4), ('n_root_samples', 200)],
            [('n_districts', 13), ('population_tolerance', .01), ('n_samples', 4), ('n_root_samples', 150)],
            [('n_districts', 20), ('population_tolerance', .02), ('n_samples', 3), ('n_root_samples', 100)],
            [('n_districts', 40), ('population_tolerance', .025), ('n_samples', 3), ('n_root_samples', 60)],
            [('n_districts', 50), ('population_tolerance', .03), ('n_samples', 3), ('n_root_samples', 50)],
            [('n_districts', 60), ('population_tolerance', .035), ('n_samples', 3), ('n_root_samples', 40)],
            [('n_districts', 80), ('population_tolerance', .04), ('n_samples', 2.4), ('n_root_samples', 30)],
            [('n_districts', 100), ('population_tolerance', .045), ('n_samples', 2.1), ('n_root_samples', 50)],
            [('n_districts', 120), ('population_tolerance', .05), ('n_samples', 2), ('n_root_samples', 40)],
            [('n_districts', 140), ('population_tolerance', .05), ('n_samples', 2), ('n_root_samples', 30)],
        ]
    }

    experiment = Experiment(base_config, experiment_config)
    experiment.run()

