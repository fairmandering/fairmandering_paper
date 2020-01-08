import networkx as nx
import pandas as pd
import numpy as np
from gerrypy.data.synmap import generate_map
from gerrypy.optimize.prune import make_lengths_data
from gerrypy.optimize.initial_cols import *

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
        state_df = pd.DataFrame({'population': pop_array,
                                 'affiliation': pop_array,
                                 'x': x,
                                 'y': y})
        lengths = make_lengths_data(config, state_df)
    else:
        pass

    solution_logger = build_initial_columns(config, state_df, lengths)
    clean_cols, cleaning_results = clean_initial_cols(G, solution_logger)
    removal_df = analyze_initial_cols(config, solution_logger, cleaning_results)
    print(removal_df)
    return clean_cols


if __name__ == '__main__':
    synmap_config = {
        'n_cities': 10,
        'exponential_scale': 1.5,
        'dem_vote': .5,
        'height': 15,
        'width': 30,
        'sigma_bounds': (.2, .6),
        'scale_bounds': (.2, .5),
        'rural_vote_std': .03
    }

    init_config = {
        # List where l[i] indicates number of trials with i random seeds
        'n_random_seeded_kmeans_iters': [1, 3, 3],
        'n_random_seeded_kmedians_iters': [1, 1, 1],
        'n_barrier_sample_iters': 5,
        'center_pruning_percent': .75,
        'population_pruning_radius': 8
    }

    main_loop_config = {

    }

    config = {
        'euclidean': True,
        'n_districts': 6,
        'population_tolerance': .05,
        'cost_exponential': 1,
        'spectral_activation_threshold': 1e-4,
        'barrier_timeout': 100,  # Seconds
        'barrier_convergence_tol': 1e-2,
        'IP_gap_tol': 1e-2,
        'IP_timeout': 5,
        'synmap_config': synmap_config,
        'init_config': init_config
    }
    solve(config)