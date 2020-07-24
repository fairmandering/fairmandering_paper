import pickle
import os
import time
from gerrypy.optimize import generate
from gerrypy import constants as consts
from scipy.spatial.distance import pdist, squareform
from gerrypy.optimize.master import make_master
import gpytorch
from gerrypy.gp import exact
from gerrypy.analyze.districts import *
from gerrypy.analyze.plan import *
from gerrypy.data.load import load_opt_data
from scipy.stats import norm

def make_train_df(district_df, tpoint=3000):
    train_cols = ['population', 'p_public_transportation_commute', 'p_walk_commute',
       'mean_commute_time', 'p_bachelors_degree_or_higher',
       'unemployment_rate', 'p_GRAPI<15%', 'p_GRAPI>35%',
       'p_without_health_insurance', 'p_nonfamily_household',
       'p_vacant_housing_units', 'p_renter_occupied',
       'median_household_income', 'p_SNAP_benefits', 'p_below_poverty_line',
       'p_white', 'p_age_students', 'median_age', 'p_mobile_homes',
       'p_without_person_vehicle', 'p_veterans', 'label', 'avg_past']
    tpoint = min(tpoint, len(district_df))
    sample_ix = np.random.choice(len(district_df), size=tpoint, replace=False)
    sample_year = np.random.choice(np.array(['2008', '2012', '2016']), size=tpoint)
    sample_avg1 = np.array([str(int(y) - 4) for y in sample_year])
    sample_avg2 = np.array([str(int(y) - 8) for y in sample_year])

    train_df = district_df.iloc[sample_ix]
    train_df['avg_past'] = train_df.lookup(train_df.index, sample_avg1) + train_df.lookup(train_df.index, sample_avg2) / 2
    train_df['label'] = train_df.lookup(train_df.index, sample_year)
    return train_df[train_cols]


def state_baseline_results(district_df):
    mean = district_df[['2008', '2012', '2016']].mean(axis=1).values
    std = district_df[['2008', '2012', '2016']].std(axis=1).values * 3 / 2 / 3 ** .5
    # std = np.maximum(std, np.median(std))
    ub = mean + std
    lb = mean - std
    pred = mean
    pred_ub = ub
    pred_lb = lb

    return pred, pred_lb, pred_ub


def state_gp_results(district_df):
    train_df = make_train_df(district_df)
    train_df['year'] = 2016
    train_df = train_df.set_index('year', append=True)
    district_df['avg_past'] = district_df[['2008', '2012', '2016']].mean(axis=1)
    district_df['label'] = np.nan
    district_df['year'] = 2016
    district_df = district_df.set_index('year', append=True)
    print(len(district_df))
    test_df = district_df[train_df.columns]
    prepro = exact.preprocess_input(train_df, test_df,
                                    normalize_per_year=True,
                                    normalize_labels=True,
                                    use_boxcox=False)
    train_x, test_x, train_y, test_y, label_scaler = prepro
    print(test_x.shape)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    kernel = gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1])
    model = exact.ExactGPModel(train_x,
                               train_y,
                               likelihood,
                               kernel)

    model, likelihood = exact.train(model,
                                    likelihood,
                                    train_x,
                                    train_y,
                                    lr=.1,
                                    training_iterations=50)

    gp_mean, gp_std = exact.evaluate(model, likelihood, test_x)

    ub = gp_mean + gp_std
    lb = gp_mean - gp_std
    pred = label_scaler.inverse_transform(gp_mean.reshape(-1, 1)).flatten()
    pred_ub = label_scaler.inverse_transform(ub.reshape(-1, 1)).flatten()
    pred_lb = label_scaler.inverse_transform(lb.reshape(-1, 1)).flatten()

    return pred, pred_lb, pred_ub


def solve(config):
    state_fips = consts.ABBREV_DICT[config['state']][consts.FIPS_IX]
    data_path = os.path.join(consts.OPT_DATA_PATH, str(state_fips))
    real_input = load_opt_data(data_path)
    state_df, G, state_covar, lengths = real_input
    gen_t_start = time.time()
    if config['saved_districts_path']:
        block_district_matrix = np.load(config['saved_districts_path'],
                                        allow_pickle=True)[()]['block_district_matrix']
        district_blocks = [np.nonzero(row)[0] for row in block_district_matrix.T]
    else:
        cg = generate.ColumnGenerator(config, config['state'])
        cg.generate()
        district_blocks = [d.area for d in cg.leaf_nodes]
        block_district_matrix = np.zeros((len(state_df), len(district_blocks)))
        for ix, d in enumerate(district_blocks):
            block_district_matrix[d, ix] = 1

    gen_t = time.time() - gen_t_start
    district_df = create_district_df(block_district_matrix, state_df)
    mean, pred_lb, pred_ub = state_gp_results(district_df)
    std = (pred_ub - pred_lb) / 2

    p_win = 1 - norm.cdf(.5, loc=mean, scale=std)
    expected_diff = p_win - mean

    district_dispersion = dispersion_compactness(district_blocks, state_df)
    district_roeck = 1 - np.array(
        roeck_compactness(district_blocks, state_df, squareform(pdist(state_df[['x', 'y']].values))))

    results = {}
    opt_time = {}
    # Compactness
    m, x = make_master(config['n_districts'],
                       block_district_matrix,
                       district_roeck,
                       opt_type='minimize')
    m.Params.MIPGapAbs = config['master_abs_gap']
    opt_start = time.time()
    m.optimize()
    opt_time['compactness'] = time.time() - opt_start
    distrs = [j for j, v in x.items() if v.X > .5]
    results['compactness'] = {d: list(district_blocks[d]) for ix, d in enumerate(distrs)}

    # Fairness
    m, x = make_master(config['n_districts'],
                       block_district_matrix,
                       expected_diff,
                       opt_type='abs_val')
    m.Params.MIPGapAbs = config['master_abs_gap']
    opt_start = time.time()
    m.optimize()
    opt_time['fairness'] = time.time() - opt_start
    distrs = [j for j, v in x.items() if v.X > .5]
    results['fairness'] = {d: list(district_blocks[d]) for ix, d in enumerate(distrs)}

    # Dem advantage
    m, x = make_master(config['n_districts'],
                       block_district_matrix,
                       expected_diff,
                       opt_type='minimize')
    m.Params.MIPGapAbs = config['master_abs_gap']
    opt_start = time.time()
    m.optimize()
    opt_time['dem_advantage'] = time.time() - opt_start
    distrs = [j for j, v in x.items() if v.X > .5]
    results['dem_advantage'] = {d: list(district_blocks[d]) for ix, d in enumerate(distrs)}

    # Rep advantage
    m, x = make_master(config['n_districts'],
                       block_district_matrix,
                       expected_diff,
                       opt_type='maximize')
    m.Params.MIPGapAbs = config['master_abs_gap']
    opt_start = time.time()
    m.optimize()
    opt_time['rep_advantage'] = time.time() - opt_start
    distrs = [j for j, v in x.items() if v.X > .5]
    results['rep_advantage'] = {d: list(district_blocks[d]) for ix, d in enumerate(distrs)}


    opt_results = {}
    for k, cols in results.items():
        opt_results[k] = {
            'dispersion': np.array(district_dispersion)[list(cols.keys())].mean(),
            'roeck': np.abs(district_roeck - 1)[list(cols.keys())].mean(),
            'comp': competitiveness(district_df.loc[cols], threshold=.05),
            'eg': efficiency_gap(district_df.loc[cols]),
            'ess': historical_seat_share(district_df.loc[cols]),
            'gp_obj': expected_diff[list(cols.keys())].sum(),
            'runtime': opt_time[k],
            'gen_time': gen_t
        }

    return results, opt_results, district_df


if __name__ == '__main__':
    center_selection_config = {
        'selection_method': 'uncapacitated_kmeans',  # one of
        'center_assignment_order': 'descending',
        'perturbation_scale': 1,
        'n_random_seeds': 1,
        'capacity_kwargs': {'weights': 'voronoi', 'capacities': 'match'}
    }

    config = {
        'n_districts': 13,
        'enforce_connectivity': True,
        'population_tolerance': .01,
        'center_selection_config': center_selection_config,
        'master_abs_gap': 1e-3,
        'master_max_time': 5,
        'IP_gap_tol': 1e-4,
        'IP_timeout': 10,
        'event_logging': False,
        'verbose': False,
        'max_sample_tries': 25,
        'n_samples': 5,
        'n_root_samples': 10,
        'max_n_splits': 5,
        'min_n_splits': 2,
        'max_split_population_difference': 1.5,
        'state': 'NC',
        'saved_districts_path': None,
    }
    n_trials = 10
    save_dir = 'NC_generation_results_2'
    try:
        os.mkdir(save_dir)
    except:
        pass
    for i in range(n_trials):
        results, opt_results, district_df = solve(config)
        pickle.dump([results, opt_results, district_df],
                    open(os.path.join(save_dir, 'opttrial%d.p' % i), 'wb'))
