import networkx as nx
import pandas as pd
import numpy as np
from gerrypy.optimize.cost import expected_rep_gap
from gerrypy.data.synmap import generate_map
from gerrypy.data.synpolitics import create_political_distribution
from gerrypy.optimize.prune import make_lengths_data, complete_lengths_data
from gerrypy.optimize.initial_cols import *
from gerrypy.optimize.problems.master import make_master
from gerrypy.optimize.hierarchical import non_binary_bfs_split
from gerrypy.optimize.tree import SampleTree
from gurobipy import *
import matplotlib.pyplot as plt


def solve(config, data_paths=None):
    if data_paths is None:
        synmap_config = config['synmap_config']
        syn_map = generate_map(synmap_config)
        h, w = synmap_config['height'], synmap_config['width']
        G = nx.grid_graph([w, h])
        G = nx.convert_node_labels_to_integers(G)
        pop_array = syn_map.flatten() * 100
        x = np.arange(w).repeat(h).reshape(w, h).T.flatten()
        y = np.arange(h).repeat(w).reshape(h, w).flatten()
        z = np.zeros(h * w)
        state_df = pd.DataFrame({'population': pop_array,
                                 'x': x, 'y': y, 'z': z})
        state_poli_mean, state_covar = create_political_distribution(config,
                                                                     state_df)
        state_df['affiliation'] = state_poli_mean
        # lengths = make_lengths_data(config, state_df)
        lengths = complete_lengths_data(state_df)
    else:
        pass

    # solution_logger = build_initial_columns(config, state_df, lengths)
    # clean_cols, cleaning_results = clean_initial_cols(G, solution_logger)
    # try:
    #     removal_df = analyze_initial_cols(config, solution_logger, cleaning_results)
    #     print(removal_df)
    # except KeyError:
    #     print('Removal Analysis Failed')
    all_cols = []
    all_costs = []
    maps = []

    # Initialization iteration
    tree = SampleTree(config['hconfig'], config['n_districts'])
    clean_cols = non_binary_bfs_split(config, G, state_df, lengths, tree)
    all_cols += clean_cols

    costs = [expected_rep_gap(distr,
                              state_df.population.values,
                              state_df.affiliation.values,
                              state_covar) for distr in clean_cols]

    all_costs += costs
    master, variables = make_master(config['n_districts'], len(state_df),
                                    all_cols, all_costs, relax=False)
    master.update()
    master.optimize()

    master_constraints = master.getConstrs()

    for i in range(1):
        tree = SampleTree(config['hconfig'], config['n_districts'])
        clean_cols = non_binary_bfs_split(config, G, state_df, lengths, tree)
        all_cols += clean_cols
        costs = [expected_rep_gap(distr,
                                  state_df.population.values,
                                  state_df.affiliation.values,
                                  state_covar) for distr in clean_cols]

        all_costs += costs
        for col, cost in zip(clean_cols, costs):
            master_col = Column()
            # Tract membership terms
            master_col.addTerms(np.ones(len(col)),
                                [master_constraints[i] for i in col])
            # abs value +, abs value -, n_districts
            control_terms = [cost, -cost, 1]
            master_col.addTerms(control_terms, master_constraints[-3:])
            var_num = len(variables)
            variables[var_num] = master.addVar(vtype=GRB.BINARY,
                                               name="x(%s)" % var_num,
                                               column=master_col,
                                               obj=cost)

        master.update()
        master.optimize()

        acts = [v.X for a, v in variables.items()]
        distrs = [i for i, v in enumerate(acts) if v > .5]
        maps.append({d: all_cols[d] for d in distrs})

        print('State Affiliation',
              state_df.affiliation.values.dot(state_df.population.values)
              / state_df.population.values.sum())
        for d in distrs:
            district = all_cols[d]
            pop_array = state_df.population.values[district]
            print('Population:', round(pop_array.sum()),
                  '  | Political Affiliation:',
                  state_df.affiliation.values[district].dot(pop_array)
                  / pop_array.sum())

    return maps


if __name__ == '__main__':
    synmap_config = {
        'n_cities': 10,
        'exponential_scale': 1.5,
        'dem_vote': .5,
        'height': 25,
        'width': 40,
        'sigma_bounds': (.2, .6),
        'scale_bounds': (.2, .5),
        'rural_vote_std': .03
    }

    politics_config = {
        'noise_mean': 0,
        'noise_variance': .1,
        'covariance_gamma': .3
    }

    init_config = {
        # List where l[i] indicates number of trials with i random seeds
        'n_random_seeded_kmeans_iters': [3, 10, 10],
        'n_random_seeded_kmedians_iters': [1, 2, 2],
        'n_barrier_sample_iters': 5,
        'center_pruning_percent': .75,
        'population_pruning_radius': 8
    }

    hconfig = {
        # Number of attempts to sample a node
        'n_sample_tries': 4,
        'n_samples': 3,
        'max_n_splits': 5,
        'min_n_splits': 2,
        'max_split_population_difference': 1.5,
    }

    config = {
        'n_districts': 11,
        'cost_exponential': 1,
        'population_tolerance': .05,
        'spectral_activation_threshold': 1e-4,
        'barrier_timeout': 100,  # Seconds
        'barrier_convergence_tol': 1e-2,
        'IP_gap_tol': 1e-2,
        'IP_timeout': 5,
        'synmap_config': synmap_config,
        'init_config': init_config,
        'politics_config': politics_config,
        'hconfig': hconfig
    }
    solve(config)
