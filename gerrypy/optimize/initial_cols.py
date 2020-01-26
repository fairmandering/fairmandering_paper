import time
from pandas import DataFrame
from gerrypy.optimize.problems.transportation import make_transportation_problem
from gerrypy.optimize.problems.kmedians import make_kmedians
from gerrypy.optimize.kmeans import *
from gerrypy.optimize.prune import yi_prune
from gerrypy.optimize.spectral import *


def gen_kmeans_cols(config, state_df, lengths, random_seeds=0):
    iter_start = time.time()
    centers = euclidean_kmeans_seeds(config, state_df, random_seeds)
    kmeans_finish_t = time.time()
    tp_lengths = {i: lengths[i] for i in centers}
    tp, xs = make_transportation_problem(config, tp_lengths,
                                         state_df.population.to_dict())
    setup_finish_t = time.time()
    tp.optimize()
    opt_finish_t = time.time()
    districting = {i: [j for j in xs[i] if xs[i][j].X > .5]
                   for i in centers}
    sol_dict = {
        'solution': districting,
        'sol_type': 'kmeans',
        'random_seeds': random_seeds,
        'center_selection_t': kmeans_finish_t - iter_start,
        'setup_t': setup_finish_t - kmeans_finish_t,
        'opt_t': opt_finish_t - setup_finish_t
    }
    return sol_dict


def gen_tp_cols(config, state_df, lengths,
                xij_activation, y_activation, method):
    sol_dict = {}
    spec_start_t = time.time()
    cluster_ys = spectral_cluster(config, xij_activation)
    centers = select_centers(state_df, cluster_ys, y_activation,
                             method=method)
    spec_end_t = time.time()

    IP_start_t = time.time()
    centers_dict = {i: lengths[i] for i in centers}
    transport, xs = make_transportation_problem(config,
                                                centers_dict,
                                                state_df.population.to_dict())
    transport.Params.MIPGap = config['IP_gap_tol']
    transport.Params.TimeLimit = config['IP_timeout']
    transport.optimize()
    IP_end_t = time.time()

    sol_dict['spectral_select_t'] = spec_end_t - spec_start_t
    sol_dict['IP_solve_t'] = IP_end_t - IP_start_t

    try:
        districting = {i: [j for j in xs[i] if xs[i][j].X > .5]
                       for i in centers}
        sol_dict['solution'] = districting
    except AttributeError:
        sol_dict['solution'] = None

    return sol_dict


def solve_barrier(config, state_df, lengths, random_seeds=0):
    partial_sol_dict = {}
    barrier_start_t = time.time()
    prune_percent = config['init_config']['center_pruning_percent']
    tract_dist_dict = yi_prune(lengths, prune_percent)
    pop_dict = state_df['population'].to_dict()

    model, xs, ys = make_kmedians(config, pop_dict, tract_dist_dict,
                                  random_seeds=random_seeds)
    barrier_setup_t = time.time()

    model.Params.TimeLimit = config['barrier_timeout']
    model.Params.BarConvTol = config['barrier_convergence_tol']
    model.Params.Crossover = 0
    model.Params.Method = 2
    model.optimize()
    barrier_solve_t = time.time()

    partial_sol_dict['barrier_setup_t'] = barrier_setup_t - barrier_start_t
    partial_sol_dict['barrier_solve_t'] = barrier_solve_t - barrier_setup_t
    partial_sol_dict['sol_type'] = 'spectral_selection'
    partial_sol_dict['random_seeds'] = random_seeds

    activation_threshold = config['spectral_activation_threshold']
    model.update()
    xij_activation = [('center' + str(i), str(j), {'weight': xs[i][j].X})
                      for i in xs for j in xs[i]
                      if xs[i][j].X > activation_threshold]
    y_activation = {i: ys[i].X if ys[i].X > activation_threshold
    else 0 for i in ys}

    return xij_activation, y_activation, partial_sol_dict


def build_initial_columns(config, state_df, lengths):
    sols = 0
    solution_logger = {}

    kmeans_trails = config['init_config']['n_random_seeded_kmeans_iters']
    for n_random_seeds, n_trials in enumerate(kmeans_trails):
        for _ in range(n_trials):
            sol_dict = gen_kmeans_cols(config, state_df, lengths,
                                       random_seeds=n_random_seeds)
            solution_logger[sols] = sol_dict
            sols += 1

    kmedian_trials = config['init_config']['n_random_seeded_kmedians_iters']
    for n_random_seeds, n_trials in enumerate(kmedian_trials):
        for _ in range(n_trials):
            result = solve_barrier(config, state_df, lengths,
                                   random_seeds=n_random_seeds)
            xij_activation, y_activation, partial_sol_dict = result

            for j in range(config['init_config']['n_barrier_sample_iters']):
                sol_dict = gen_tp_cols(config, state_df, lengths,
                                       xij_activation, y_activation,
                                       method='sample' if j > 0 else 'average')
                for k, v in partial_sol_dict.items():
                    sol_dict[k] = v
                solution_logger[sols] = sol_dict
                sols += 1

    return solution_logger


def clean_initial_cols(G, init_solutions):
    results = {}
    clean_cols = set()
    for i, sol_dict in init_solutions.items():
        if sol_dict['solution'] is None:
            results[i, 0] = 'infeasible'
            continue
        for center, district in sol_dict['solution'].items():
            if not nx.is_connected(G.subgraph(district)):
                results[i, center] = 'disconnected'
                continue
            district = frozenset(district)
            if district in clean_cols:
                results[i, center] = 'duplicate'
            else:
                clean_cols.add(district)
    return list([list(c) for c in clean_cols]), results


def analyze_initial_cols(config, solution_logger, cleaning_results):
    whys = []  # Why was the column removed
    whats = []  # Which method generated removed column
    whens = []  # When in the process did it happen
    rand_seeds = []
    for (when, _), why in cleaning_results.items():
        whys.append(why)
        whats.append(solution_logger[when]['sol_type'])
        rand_seeds.append(solution_logger[when]['random_seeds'])
        whens.append(when)
    removal_df = DataFrame({'why': whys,
                            'what': whats,
                            'when': whens,
                            'n_random': rand_seeds})
    n_cols_generated = config['n_districts'] * len(solution_logger)
    group_df = removal_df.groupby('why') \
                   .apply(lambda x: x.groupby('what') \
                          .apply(lambda y: y.groupby('n_random')['why'] \
                                 .count())) / n_cols_generated * 100
    return group_df
