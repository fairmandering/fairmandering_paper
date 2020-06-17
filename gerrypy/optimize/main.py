import networkx as nx
import pandas as pd
import pickle
import numpy as np
from gerrypy.optimize.cost import expected_rep_gap
from gerrypy.data.synthetic import generate_synthetic_input
from gerrypy.optimize.prune import make_lengths_data, complete_lengths_data
from gerrypy.optimize.problems.master import make_master
from gerrypy.optimize.tree import SampleTree
from gurobipy import *


def load_real_data(data_base_path):
    state_df_path = os.path.join(data_base_path, 'state_df.csv')
    adjacency_graph_path = os.path.join(data_base_path, 'G.p')

    state_df = pd.read_csv(state_df_path)
    G = nx.read_gpickle(adjacency_graph_path)

    if os.path.exists(os.path.join(data_base_path, 'lengths.npy')):
        lengths_path = os.path.join(data_base_path, 'lengths.npy')
        lengths = np.load(lengths_path)
    else:
        from scipy.spatial.distance import pdist, squareform
        lengths = squareform(pdist(state_df[['x', 'y']].values))

    if os.path.exists(os.path.join(data_base_path, 'edge_dists.p')):
        edge_dists_path = os.path.join(data_base_path, 'edge_dists.p')
        edge_dists = pickle.load(open(edge_dists_path, 'rb'))
    else:
        edge_dists = dict(nx.all_pairs_shortest_path_length(G))

    return state_df, G, lengths, edge_dists


def solve(config, data_base_path=None):
    if data_base_path is None:
        synthetic_input = generate_synthetic_input(config)
        state_df, G, state_covar, lengths = synthetic_input
    else:
        real_input = load_real_data(data_base_path)
        state_df, G, state_covar, lengths = real_input


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

    for i in range(config['n_tree_samples']):
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
            # n_districts, abs value +, abs value -
            control_terms = [1, cost, cost]
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
        'n_tree_samples': 10,
        'cost_exponential': 1,
        'population_tolerance': .05,
        'IP_gap_tol': 1e-2,
        'IP_timeout': 5,
        'master_abs_gap': 1e-3,
        'synmap_config': synmap_config,
        'politics_config': politics_config,
        'hconfig': hconfig
    }
    solve(config)
