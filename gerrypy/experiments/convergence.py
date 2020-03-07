import time
import os
import json

from gerrypy.analyze.algorithm import number_of_districtings
from gerrypy.data.synthetic import generate_synthetic_input
from gerrypy.optimize.cost import expected_rep_gap
from gerrypy.optimize.main import load_real_data
from gerrypy.optimize.shc import *
from gerrypy.optimize.problems.master import make_master


def run_experiment(config, data_base_path, n_trials, save_dir):
    if data_base_path is None:
        data_input = generate_synthetic_input(config)
    else:
        data_input = load_real_data(data_base_path)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for trial in range(n_trials):
        result = run_generation_trial(config, data_input)
        save_name = '_'.join(['trial%d' % trial,
                              'forced_connectivity' if config['enforce_connectivity'] else '',
                              str(config['hconfig']['n_samples']),
                              str(time.time()).replace('.', '')]) + '.json'
        json.dump(result, open(os.path.join(save_dir, save_name), 'w'))
        if trial % 100 == 0:
            print('Completed trial', trial)



def run_generation_trial(config, data_input):
    state_df, G, state_covar, lengths, edge_dists = data_input
    completed_trial = False
    failed_trials = 0

    start = time.time()
    while not completed_trial:
        try:
            leaves, interior = shc(config, G, state_df, lengths, edge_dists)
            completed_trial = True
            print('success')
        except RuntimeError:
            failed_trials += 1
            print('failure')

    columns = [leaf.area for leaf in leaves]
    costs = [expected_rep_gap(area,
                              state_df.population.values,
                              state_df.affiliation.values,
                              state_covar) for area in columns]
    end = time.time()

    result_dict = {
        'n_failed_trials': failed_trials,
        'elapsed_time': end - start,
        'n_leaves': len(leaves),
        'n_interior': len(interior),
        'n_failed_samples': [i.n_sample_failures for i in interior],
        'n_disconnected_samples': [i.n_disconnected_samples for i in interior],
        'n_unique_districtings': number_of_districtings(leaves, interior),
        'hconfig': config['hconfig'],
        'columns': columns,
        'costs': costs
    }

    return result_dict


def run_optimization(config, column_path, state_df):

    with open(column_path, 'r') as f:
        result_dict = json.load(f)

    all_columns = result_dict['columns']
    all_costs = result_dict['costs']

    unique_columns = set()
    columns = []
    costs = []
    for ix, col in enumerate(all_columns):
        fcol = frozenset(col)
        if fcol in unique_columns:
            continue
        else:
            columns.append(col)
            unique_columns.add(fcol)
            costs.append(all_costs[ix])

    start = time.time()
    if result_dict['n_unique_districtings'] == 1:
        cost = sum(costs)
        result_dict['opt_fair_obj'] = cost

    else:
        config['master_max_time'] *= result_dict['hconfig']['n_samples'] ** 2
        # Fairness
        master, variables = make_master(config['n_districts'], len(state_df),
                                        columns, costs, relax=False)

        master.Params.MIPGapAbs = config['master_abs_gap']
        master.Params.TimeLimit = config['master_max_time']
        master.update()
        master.optimize()
        result_dict['opt_fair_obj'] = master.objVal
        acts = [v.X for a, v in variables.items()]
        distrs = [i for i, v in enumerate(acts) if v > .5]
        districting = {d: columns[d] for d in distrs}
        result_dict['opt_fair_sol'] = districting

        # Optimized for democrats
        master, variables = make_master(config['n_districts'], len(state_df),
                                        columns, costs, relax=False,
                                        maximize_dem_advantage=True)

        master.Params.MIPGapAbs = config['master_abs_gap']
        master.Params.TimeLimit = config['master_max_time']
        master.update()
        master.optimize()
        result_dict['opt_dem_obj'] = master.objVal
        acts = [v.X for a, v in variables.items()]
        distrs = [i for i, v in enumerate(acts) if v > .5]
        districting = {d: columns[d] for d in distrs}
        result_dict['opt_dem_sol'] = districting

        # Optimized for republicans
        master, variables = make_master(config['n_districts'], len(state_df),
                                        columns, costs, relax=False,
                                        maximize_rep_advantage=True)

        master.Params.MIPGapAbs = config['master_abs_gap']
        master.Params.TimeLimit = config['master_max_time']
        master.update()
        master.optimize()
        result_dict['opt_rep_obj'] = master.objVal
        acts = [v.X for a, v in variables.items()]
        distrs = [i for i, v in enumerate(acts) if v > .5]
        districting = {d: columns[d] for d in distrs}
        result_dict['opt_rep_sol'] = districting

    end = time.time()

    result_dict['opt_time'] = end - start

    json.dump(result_dict, open(column_path, 'w'))


if __name__ == '__main__':
    hconfig = {
        # Number of attempts to sample a node
        'n_sample_tries': 20,
        'n_samples': 1,
        'max_n_splits': 5,
        'min_n_splits': 2,
        'max_split_population_difference': 1.5
    }

    config = {
        'n_districts': 27,
        'n_tree_samples': 1,
        'population_tolerance': .05,
        'master_abs_gap': 1e-3,
        'master_max_time': 5,
        'IP_gap_tol': 1e-2,
        'IP_timeout': 5,
        'enforce_connectivity': True,
        'hconfig': hconfig
    }

    data_base_path = os.path.abspath(
        r'C:\Users\6burg\Documents\gerrymandering\gerrypy\gerrypy\data\optimization_data\12')
    save_dir = os.path.abspath(r'C:\Users\6burg\Documents\gerrymandering\gerrypy\gerrypy\results\fl\columns')
    
    experiment_sample_factor = [8]
    experiment_n_trials = [20]
    
    for n_samples, n_trials in zip(experiment_sample_factor,
                                   experiment_n_trials):
        config['hconfig']['n_samples'] = n_samples
        run_experiment(config, data_base_path, n_trials, save_dir)
