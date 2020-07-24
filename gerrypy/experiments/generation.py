from gerrypy.optimize.generate import ColumnGenerator
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
        save_dir = '%s_results_%s' % (name, str(int(time.time())))
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

                print('Starting trial', trial_config)
                cg = ColumnGenerator(trial_config, state)
                generation_start_t = time.time()
                cg.generate()
                generation_t = time.time() - generation_start_t
                analysis_start_t = time.time()
                metrics, sigma, bdm = cg.district_metrics()
                analysis_t = time.time() - analysis_start_t

                trial_results = {
                    'generation_time': generation_t,
                    'analysis_time': analysis_t,
                    'cg_metrics': metrics,
                    'singular_values': sigma,
                    'block_district_matrix': bdm,
                    'n_unique_districtings': cg.number_of_districtings(),
                    'leaf_nodes': cg.leaf_nodes,
                    'internal_nodes': cg.internal_nodes,
                    'trial_config': trial_config,
                    'trial_values': trial_values
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


if __name__ == '__main__':
    center_selection_config = {
        'selection_method': 'uncapacitated_kmeans',  # one of
        'center_assignment_order': 'descending',
        'perturbation_scale': 1,
        'n_random_seeds': 1,
        'capacity_kwargs': {'weights': 'voronoi', 'capacities': 'match'}
    }

    base_config = {
        'n_districts': 18,
        'enforce_connectivity': True,
        'population_tolerance': .01,
        'center_selection_config': center_selection_config,
        'master_abs_gap': 1e-3,
        'master_max_time': 5,
        'IP_gap_tol': 1e-4,
        'IP_timeout': 10,
        'event_logging': False,
        'verbose': False,
        'max_sample_tries': 30,
        'n_samples': 7,
        'n_root_samples': 10,
        'max_n_splits': 5,
        'min_n_splits': 2,
        'max_split_population_difference': 1.5
    }
    experiment_config = {
        'name': 'IL_opt_cols',
        'states': ['IL'],
        'trial_parameters': [
            [('n_samples', 7)],
            [(('center_selection_config', 'capacity_kwargs'),
              {'weights': 'voronoi', 'capacities': 'compute'})],
        ]
    }

    experiment = Experiment(base_config, experiment_config)
    experiment.run()

