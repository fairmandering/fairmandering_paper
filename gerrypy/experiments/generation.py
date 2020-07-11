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
        'perturbation_scale': .5,
        'n_random_seeds': 0,
        'capacity_kwargs': {'weights': 'voronoi', 'capacities': 'match'}
    }

    base_config = {
        'n_districts': 18,
        'enforce_connectivity': True,
        'population_tolerance': .025,
        'center_selection_config': center_selection_config,
        'master_abs_gap': 1e-3,
        'master_max_time': 5,
        'IP_gap_tol': 1e-4,
        'IP_timeout': 10,
        'event_logging': False,
        'verbose': False,
        'max_sample_tries': 20,
        'n_samples': 3,
        'n_root_samples': 150,
        'max_n_splits': 5,
        'min_n_splits': 2,
        'max_split_population_difference': 1.5
    }
    experiment_config = {
        'name': 'IL_frac_results',
        'states': ['IL'],
        'trial_parameters': [
             [(('center_selection_config', 'capacity_kwargs'),
               {'weights': 'fractional', 'dist_penalty': 2, 'capacities': 'compute'}),
              (('center_selection_config', 'n_random_seeds'), 1),
              (('center_selection_config', 'selection_method'), 'uncapacitated_kmeans')],
             [(('center_selection_config', 'capacity_kwargs'),
               {'weights': 'fractional', 'dist_penalty': 2, 'capacities': 'compute'}),
              (('center_selection_config', 'center_assignment_order'), 'random'),
              (('center_selection_config', 'selection_method'), 'random_iterative')],
             [(('center_selection_config', 'capacity_kwargs'),
               {'weights': 'fractional', 'dist_penalty': 2, 'capacities': 'match'}),
              (('center_selection_config', 'perturbation_scale'), 0.5),
              (('center_selection_config', 'selection_method'), 'uncapacitated_kmeans')],
             [(('center_selection_config', 'capacity_kwargs'),
               {'weights': 'fractional', 'dist_penalty': 2, 'capacities': 'match'}),
              (('center_selection_config', 'perturbation_scale'), 1),
              (('center_selection_config', 'selection_method'), 'uncapacitated_kmeans')],
             [(('center_selection_config', 'capacity_kwargs'),
               {'weights': 'fractional', 'dist_penalty': 2, 'capacities': 'match'}),
              (('center_selection_config', 'n_random_seeds'), 1),
              (('center_selection_config', 'selection_method'), 'uncapacitated_kmeans')],
             [(('center_selection_config', 'capacity_kwargs'),
               {'weights': 'fractional', 'dist_penalty': 2, 'capacities': 'match'}),
              (('center_selection_config', 'center_assignment_order'), 'random'),
              (('center_selection_config', 'selection_method'), 'random_iterative')]
        ]
    }

    experiment = Experiment(base_config, experiment_config)
    experiment.run()

# Capacity matching
# [(('center_selection_config', 'capacity_kwargs'),
#   {'weights': 'fractional', 'dist_penalty': 1, 'capacities': 'compute', 'capacity_constraint': 1.5})],
# [(('center_selection_config', 'capacity_kwargs'),
#   {'weights': 'fractional', 'dist_penalty': 2, 'capacities': 'compute'})],
# [(('center_selection_config', 'capacity_kwargs'),
#   {'weights': 'voronoi', 'dist_penalty': 1, 'capacities': 'match'})],
# [(('center_selection_config', 'capacity_kwargs'),
#   {'weights': 'fractional', 'dist_penalty': 1, 'capacities': 'match'})],
# [(('center_selection_config', 'capacity_kwargs'),
#   {'weights': 'fractional', 'dist_penalty': 2, 'capacities': 'compute', 'capacity_constraint': 1.5})],
# [(('center_selection_config', 'capacity_kwargs'),
#   {'weights': 'voronoi', 'dist_penalty': 1, 'capacities': 'compute'})],
# [(('center_selection_config', 'capacity_kwargs'),
#   {'weights': 'fractional', 'dist_penalty': 1, 'capacities': 'compute'})],
# [(('center_selection_config', 'capacity_kwargs'),
#   {'weights': 'fractional', 'dist_penalty': 2, 'capacities': 'match'})],
# [(('center_selection_config', 'capacity_kwargs'),
#   {'weights': 'voronoi', 'dist_penalty': 1, 'capacities': 'compute', 'capacity_constraint': 1.5})],

# Center selection sweep
# [(('center_selection_config', 'perturbation_scale'), .25),
#  (('center_selection_config', 'selection_method'), 'uncapacitated_kmeans')],
# [(('center_selection_config', 'perturbation_scale'), .5),
#  (('center_selection_config', 'selection_method'), 'uncapacitated_kmeans')],
# [(('center_selection_config', 'perturbation_scale'), .75),
#  (('center_selection_config', 'selection_method'), 'uncapacitated_kmeans')],
# [(('center_selection_config', 'perturbation_scale'), 1),
#  (('center_selection_config', 'selection_method'), 'uncapacitated_kmeans')],
# [(('center_selection_config', 'perturbation_scale'), 1.5),
#  (('center_selection_config', 'selection_method'), 'uncapacitated_kmeans')],
# [(('center_selection_config', 'perturbation_scale'), 2),
#  (('center_selection_config', 'selection_method'), 'uncapacitated_kmeans')],
#
# [(('center_selection_config', 'n_random_seeds'), .25),
#  (('center_selection_config', 'selection_method'), 'uncapacitated_kmeans')],
# [(('center_selection_config', 'n_random_seeds'), .5),
#  (('center_selection_config', 'selection_method'), 'uncapacitated_kmeans')],
# [(('center_selection_config', 'n_random_seeds'), .75),
#  (('center_selection_config', 'selection_method'), 'uncapacitated_kmeans')],
# [(('center_selection_config', 'n_random_seeds'), 1),
#  (('center_selection_config', 'selection_method'), 'uncapacitated_kmeans')],
# [(('center_selection_config', 'n_random_seeds'), 1.25),
#  (('center_selection_config', 'selection_method'), 'uncapacitated_kmeans')],
# [(('center_selection_config', 'n_random_seeds'), 1.5),
#  (('center_selection_config', 'selection_method'), 'uncapacitated_kmeans')],
#
# [(('center_selection_config', 'center_assignment_order'), 'ascending'),
#  (('center_selection_config', 'selection_method'), 'random_iterative')],
# [(('center_selection_config', 'center_assignment_order'), 'descending'),
#  (('center_selection_config', 'selection_method'), 'random_iterative')],
# [(('center_selection_config', 'center_assignment_order'), 'random'),
#  (('center_selection_config', 'selection_method'), 'random_iterative')],
