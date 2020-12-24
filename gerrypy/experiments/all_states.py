import os
import pickle
import time
import copy
from gerrypy import constants
from gerrypy.analyze.districts import number_of_districtings
from gerrypy.optimize.generate import ColumnGenerator
from gerrypy.pipelines import pnas
from gerrypy.utils import dir_processing


class StateRunner:
    """Experiment class to run generation on all states, generate district dfs
    and run MSP optimization on the ensemble"""
    def __init__(self, name, base_config):
        self.base_config = base_config
        self.name = name

    def k_to_w(self, k):
        """Compute the sample width as a function of the number of districts."""
        w_root = int(round((2000 / k) ** 1.2))
        w_internal = int(round((200 / k) ** .7))
        return w_root, w_internal

    def run(self, states=None, test=False, test_sample_widths=(1, 1)):
        """
        Perform the generation, save the tree and the configuration used for generation.
        Args:
            states: optional (list) of states to run generation on instead of all multi district states.
            test: (bool) whether to run a small trial run.
            test_sample_widths: (tuple) the size of the test run if [test]

        Saves three files for each state: the ensemble file containing the columns
        and generation config, the district_df csv containing district level metrics,
        and a results file of the result of the fairness optimization and other
        ensemble results.
        """
        experiment_dir_name = os.path.join(constants.RESULTS_PATH,
                                           self.name + str(int(time.time())))
        os.makedirs(experiment_dir_name)

        if states is None:
            trials = [(state, seats_dict['house']) for state, seats_dict
                      in constants.seats.items() if seats_dict['house'] > 1]
        else:
            trials = [(state, constants.seats[state]['house'])
                      for state in states]

        for state, n_seats in trials:
            if test:
                w_root, w_internal = test_sample_widths
            else:
                w_root, w_internal = self.k_to_w(n_seats)
            trial_config = copy.deepcopy(self.base_config)
            trial_config['state'] = state
            trial_config['n_districts'] = n_seats
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

        print('Finished generation...')
        print('Creating district dataframes...')
        dir_processing.district_df_of_tree_dir(experiment_dir_name)
        print('Running PNAS pipeline...')
        pnas.run_all_states_result_pipeline(experiment_dir_name,
                                            [s[0] for s in trials])


if __name__ == '__main__':
    center_selection_config = {
        'selection_method': 'uniform_random',  # one of
        'perturbation_scale': 0,
        'n_random_seeds': 0,
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
    experiment = StateRunner('uniform_random_columns', base_config)

    states = ['VA', 'WA', 'NJ', 'SC', 'FL', 'OR', 'GA', 'MS', 'AR', 'NE']
    experiment.run(states=states)
