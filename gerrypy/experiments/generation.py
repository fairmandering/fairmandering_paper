from gerrypy.optimize.partition import ColumnGenerator
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
        save_dir = 'experiment_results_%s' % str(time.time())
        os.mkdir(save_dir)
        for state in self.experiment_config['states']:
            print('############## Starting %s trials ##############' % state)
            for trial_values in self.experiment_config['trial_parameters']:
                trial_config = deepcopy(self.base_config)
                for k, v in trial_values:
                    if len(k) == 2:
                        trial_config[k[0]][k[1]] = v
                    else:
                        trial_config[k] = v

                print('Starting trial', trial_config)
                cg = ColumnGenerator(trial_config, state)
                generation_start_t = time.time()
                cg.generate()
                generation_t = time.time() - generation_start_t
                analysis_start_t = time.time()
                metrics, sigma, pdm = cg.district_metrics()
                analysis_t = time.time() - analysis_start_t

                trial_results = {
                    'generation_time': generation_t,
                    'analysis_time': analysis_t,
                    'cg_metrics': metrics,
                    'singular_values': sigma,
                    'precinct_district_matrix': pdm,
                    'n_unique_districtings': cg.number_of_districtings(),
                    'trial_config': trial_config
                }

                config_str = '_'.join([str(v) for k, v in trial_values])
                save_name = '_'.join([state, config_str, str(int(time.time() // 1))]) + '.npy'
                #json.dump(trial_results, open(os.path.join(save_dir, save_name), 'w'))
                np.save(os.path.join(save_dir, save_name), trial_results)

if __name__ == '__main__':
    center_selection_config = {
        'selection_method': 'uncapacitated_kmeans',  # one of
        'center_assignment_order': 'descending',
        'seed_to_center_method': 'identity',
        'perturbation_scale': 0,
        'n_random_seeds': 1
    }

    base_config = {
        'n_districts': 9,
        'enforce_connectivity': True,
        'population_tolerance': .05,
        'center_selection_config': center_selection_config,
        'master_abs_gap': 1e-3,
        'master_max_time': 5,
        'IP_gap_tol': 1e-4,
        'IP_timeout': 10,
        'event_logging': False,
        'verbose': False,
        'max_sample_tries': 5,
        'n_samples': 3,
        'n_root_samples': 10,
        'max_n_splits': 5,
        'min_n_splits': 2,
        'max_split_population_difference': 1.5
    }
    experiment_config = {
        'states': ['MA'],
        'trial_parameters': [
            [(('center_selection_config', 'n_random_seeds'), .25)],
            [(('center_selection_config', 'n_random_seeds'), .5)],
            [(('center_selection_config', 'n_random_seeds'), .75)],
            [(('center_selection_config', 'n_random_seeds'), 1)],
            [(('center_selection_config', 'n_random_seeds'), 1.25)],
            [(('center_selection_config', 'n_random_seeds'), 1.5)],
        ]
    }

    experiment = Experiment(base_config, experiment_config)
    experiment.run()
