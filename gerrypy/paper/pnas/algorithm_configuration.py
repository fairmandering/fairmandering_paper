import seaborn as sns

from gerrypy.analyze.subsample import *
from gerrypy.analyze.viz import *


def load_trials_df(result_path):
    """Create results dataframe from raw generation results."""
    trial_files = os.listdir(result_path)
    trials = {}
    for f in trial_files:
        if f[-2:] != '.p' and f[-4:] != '.npy':
            continue
        print(f)
        t = np.load(os.path.join(result_path, f), allow_pickle=True)[()]
        r_dict = {
            'generation_time': t['generation_time'],
            'analysis_time': t['analysis_time'],
            'n_unique_districtings': t['n_plans'],
            'name': f
        }
        config = t['trial_config']
        r = {**config, **r_dict, **t['metrics']}
        trials[f] = r
    return pd.DataFrame(trials).T


def get_center_method(row):
    """Convert center selection full name in short name."""
    if row['selection_method'] == 'random_iterative':
        return 'RI'
    elif row['selection_method'] == 'uniform_random':
        return 'UR'
    else:
        if row['n_random_seeds'] == 1 and row['perturbation_scale'] == 0:
            return 'FC'
        elif row['n_random_seeds'] == 0 and row['perturbation_scale'] > 0:
            return '%s-P' % str(row['perturbation_scale'])
        else:
            return 'FC+%s-P' % str(row['perturbation_scale'])


def get_capacity_method(row):
    """Convert capacity method full names into short names."""
    name_map = {
        'voronoi': 'V',
        'compute': 'C',
        'fractional': 'F',
        'match': 'M'
    }
    weights = name_map[row['weights']]
    cap = name_map[row['capacities']]
    return weights + '+' + cap


def process_state_trial_df(df, index_columns):
    """Format results dataframe for paper presentation."""
    drop_columns = [col for col in df.columns if len(df[col].unique()) == 1]
    drop_columns += ['name', 'n_root_failures', 'n_interior_nodes', 'max_pop_variation', 'ideal_pop', 'seat_difference']
    percent_columns = ['p_infeasible', 'p_duplicates']
    float_cols = ['generation_time', 'analysis_time', 'p_infeasible', 'p_duplicates',
                  'conditional_entropy', 'average_district_sim', '50p_approx_rank', '95p_approx_rank',
                  '99p_approx_rank', 'lambda_2', 'lambda_k', 'dispersion', 'roeck', 'n_unique_districtings']
    int_cols = ['n_interior_nodes', 'n_districts']
    df = df.drop(columns=[c for c in drop_columns if c in set(list(df.columns))])
    df = df.astype({c: float for c in float_cols if c in set(list(df.columns))})
    df = df.astype({c: np.int64 for c in int_cols if c in set(list(df.columns))})
    df['leverage'] = np.log(df['n_unique_districtings'] / df['n_districts']) / np.log(10)

    generation_drop = ['generation_time', 'analysis_time', 'n_unique_districtings', 'n_interior_nodes',
                       'n_districts', '95p_approx_rank', 'lambda_k', 'selection_method', 'perturbation_scale',
                       'n_random_seeds',
                       'weights', 'capacities', 'pol_inner_90_range', 'comp_inner_90_var']
    try:
        df['Centers'] = df[['selection_method', 'perturbation_scale', 'n_random_seeds']].apply(
            lambda x: get_center_method(x), axis=1)
        df['Capacities'] = df[['weights', 'capacities']].apply(lambda x: get_capacity_method(x), axis=1)
    except KeyError:
        pass
    col_set = set(list(df.columns))
    df = df.drop(columns=[c for c in generation_drop if c in col_set])

    column_names = {
        'state': 'State',
        'seat_disparity': 'ESR',
        'compactness_disparity': 'MCD',
        'p_infeasible': '\%infeas',
        'p_duplicates': '\%dup',
        'conditional_entropy': '$H(b_i|b_j)$',
        'average_district_sim': 'ADS',
        '50p_approx_rank': '$\sigma_{50}$',
        '99p_approx_rank': '$\sigma_{99}$',
        'lambda_2': '$\lambda_2$',
        'dispersion': '$\mu_{walk}$',
        'roeck': 'Roeck',
        'leverage': '$\ell$',
        'population_tolerance': '$\epsilon_p$'
    }
    df = df.rename(columns=column_names)

    df = df.set_index(index_columns)
    percent_columns = ['\%infeas', '\%dup']
    df[percent_columns] = df[percent_columns] * 100
    df['Roeck'] /= 1000000
    sig_difs = {
        'ESR': 3,
        'MCD': 3,
        '\\%infeas': 2,
        '\\%dup': 2,
        '$H(b_i|b_j)$': 4,
        'ADS': 3,
        '$\\sigma_{50}$': 3,
        '$\\sigma_{99}$': 3,
        '$\\lambda_2$': 3,
        '$\mu_{walk}$': 2,
        '$\ell$': 2,
        'Roeck': 3,
    }
    for k, v in sig_difs.items():
        df[k] = df[k].astype(float).round(v)

    column_order = ['$\ell$', '\\%infeas',
                    '\\%dup', '$H(b_i|b_j)$', 'ADS', '$\\sigma_{50}$', '$\\sigma_{99}$', '$\\lambda_2$',
                    '$\mu_{walk}$', 'Roeck', 'MCD', 'ESR']

    return df[column_order].sort_index(level=0)


def load_seat_distribution_by_epsilon(path):
    """Load seat distributions for varying population tolerance experiment."""
    distributions = {'IL': {}, 'NC': {}}
    for file in os.listdir(path):
        if os.path.isdir(os.path.join(path, file)):
            continue
        tree_data = np.load(os.path.join(path, file), allow_pickle=True)[()]
        leaf_nodes = tree_data['leaf_nodes']
        internal_nodes = tree_data['internal_nodes']
        ddf = pd.read_csv(os.path.join(path, 'district_dfs', file[:-4] + '_district_df.csv'))
        leaf_values = party_advantage_query_fn(ddf)
        seat_distribution = enumerate_distribution(leaf_nodes, internal_nodes, leaf_values)
        state = file.split('_')[0]
        pop_tol = file.split('_')[1]
        distributions[state][pop_tol] = seat_distribution
        print(file)
    return distributions


def plot_il_seat_distributions_varying_epislon(fig_folder, distributions):
    """Plot Illinois seat-share distribution for different population tolerances."""
    plt.rcParams.update({'font.size': 10})
    for k, d in distributions['IL'].items():
        sns.distplot(np.array(d) / constants.seats['IL']['house'], hist=False, label=k)
    plt.legend(title='$\epsilon_p$')
    plt.xlabel('Expected Republican seat-share')
    plt.ylabel('density')
    plt.title("Illinois seat distribution")
    frame = plt.gca()
    frame.axes.get_yaxis().set_ticks([])
    plt.savefig(os.path.join(fig_folder, 'il_vary_epsilon.eps'),
                format='eps', bbox_inches='tight')


def plot_nc_seat_distributions_varying_epislon(fig_folder, distributions):
    """Plot North Carolina seat-share distribution for different population tolerances."""
    plt.rcParams.update({'font.size': 10})
    for k, d in distributions['NC'].items():
        sns.distplot(np.array(d) / constants.seats['NC']['house'], hist=False, label=k)
    plt.legend(title='$\epsilon_p$')
    plt.xlabel('Expected Republican seat-share')
    plt.ylabel('density')
    plt.title("North Carolina seat distribution")
    frame = plt.gca()
    frame.axes.get_yaxis().set_ticks([])
    plt.savefig(os.path.join(fig_folder, 'nc_vary_epsilon.eps'),
                format='eps', bbox_inches='tight')


def process_vary_k_trial_df(df):
    """Format results dataframe for variable number of districts experiment."""
    float_cols = ['generation_time', 'analysis_time', 'p_infeasible', 'p_duplicates',
                  'conditional_entropy', 'average_district_sim', '50p_approx_rank', '95p_approx_rank',
                  '99p_approx_rank', 'lambda_2', 'lambda_k', 'dispersion', 'roeck', 'n_unique_districtings']
    int_cols = ['n_interior_nodes', 'n_districts']
    drop_columns = [col for col in df.columns if len(df[col].unique()) == 1]
    drop_columns += ['name', 'n_root_failures', 'n_interior_nodes', 'max_pop_variation', 'ideal_pop', 'seat_difference']

    df['plans/s'] = (df['n_unique_districtings'] / df['generation_time']).map(lambda x: '%1.2E' % x)
    df['k'] = df['name'].apply(lambda x: int(x.split('_')[1]))

    k = df['k'].values
    w = df['n_samples'].values
    r_samples = df['n_root_samples'].values
    lev_ub = r_samples / w * w ** (k - 1)
    lev_lb = r_samples / w * w ** ((k - 1) / 4)

    df['lev_ub'] = np.log((lev_ub / df['n_districts']).astype(float)) / math.log(10)
    df['lev_lb'] = np.log((lev_lb / df['n_districts']).astype(float)) / math.log(10)

    df['state'] = df['name'].apply(lambda x: x.split('_')[0])
    col_set = set(list(df.columns))
    int_cols.append('k')
    df = df.astype({c: np.float64 for c in float_cols if c in col_set})
    df = df.astype({c: np.int64 for c in int_cols if c in col_set})
    df['leverage'] = np.log(df['n_unique_districtings'] / df['n_districts']) / np.log(10)

    df['generation_time'] /= 60
    df['p_duplicates'] *= 100
    df['p_infeasible'] *= 100
    df = df[['state', 'k', 'n_root_samples', 'n_samples', 'population_tolerance', 'generation_time',
             'p_duplicates', 'p_infeasible', 'lev_lb', 'leverage', 'lev_ub', 'plans/s']]
    sig_difs = {
        'generation_time': 2,
        'p_duplicates': 3,
        'p_infeasible': 3,
        'lev_lb': 3,
        'leverage': 3,
        'lev_ub': 3,
    }
    for k, v in sig_difs.items():
        df[k] = df[k].astype(float).round(v)
    df = df.rename(columns={
        'k': '$k$',
        'n_root_samples': '$w$(root)',
        'n_samples': '$w$',
        'population_tolerance': '$\epsilon_p$',
        'generation_time': 'runtime (m)',
        'p_duplicates': 'duplicates',
        'p_infeasible': 'infeasible',
        'lev_lb': '$\ell$-lb',
        'leverage': '$\ell$',
        'lev_ub': '$\ell$-ub'
    })

    df = df.set_index(['state', '$k$']).sort_index()

    return df.sort_index(level=0).sort_index(level=1)


def seat_share_with_k_distribution(result_path):
    """Compute seat-share distribution deciles for varying numbers of districts."""
    plt.rcParams.update({'font.size': 10})
    trial_files = os.listdir(result_path)
    percentiles = {}
    for f in trial_files:
        if f[-2:] != '.p' and f[-4:] != '.npy':
            continue
        t = np.load(os.path.join(result_path, f), allow_pickle=True)[()]

        district_df = pd.read_csv(os.path.join(result_path, 'district_dfs', f[:-4] + '_district_df.csv'))
        leaf_nodes = t['leaf_nodes']
        internal_nodes = t['internal_nodes']
        state = t['trial_config']['state']
        k = t['trial_config']['n_districts']

        leaf_values = party_advantage_query_fn(district_df)
        max_value, _ = query_tree(leaf_nodes, internal_nodes, leaf_values)
        min_value, _ = query_tree(leaf_nodes, internal_nodes, -leaf_values)
        min_value = -min_value
        solution_count, parent_nodes = get_node_info(leaf_nodes, internal_nodes)
        target_size = min(1000 * k ** 2, 2000000)
        pruned_internal_nodes = prune_sample_space(internal_nodes, solution_count, parent_nodes, target_size)
        distribution = enumerate_distribution(leaf_nodes, pruned_internal_nodes, leaf_values)
        distribution.append(min_value)
        distribution.append(max_value)
        distribution = np.array(distribution) / k
        probs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        percentiles[k] = [np.percentile(distribution, p) for p in probs]
    return percentiles


def plot_nc_seat_distribution_varying_k(fig_folder, nc_percentiles):
    """Plot North Carolina seat-share for varying number of districts."""
    plt.rcParams.update({'font.size': 10})
    nc_xs = np.array(sorted(list(nc_percentiles.keys())))
    nc_percentile_list = [nc_percentiles[x] for x in nc_xs]
    plot_percentiles(nc_xs, np.array(nc_percentile_list).T)
    plt.xlabel('$k$: number of seats')
    plt.ylabel('expected Republican seat-share')
    distrs = [13, 60, 120]
    for d in distrs:
        plt.axvline(x=d, linestyle='--', linewidth=1, color='black')
    plt.text(14, .65, 'US House')
    plt.text(61, .65, 'NC Senate')
    plt.text(121, .65, 'NC House')
    plt.title('North Carolina')
    plt.savefig(os.path.join(fig_folder, 'nc_vary_k.eps'), format='eps', bbox_inches='tight')


def plot_il_seat_distribution_varying_k(fig_folder, il_percentiles):
    """Plot Illinois seat-share for varying number of districts."""
    plt.rcParams.update({'font.size': 10})
    il_xs = np.array(sorted(list(il_percentiles.keys())))
    il_percentile_list = [il_percentiles[x] for x in il_xs]
    plot_percentiles(il_xs, np.array(il_percentile_list).T)
    plt.xlabel('$k$: number of seats')
    plt.ylabel('expected Republican seat-share')
    distrs = [18, 59, 118]
    for d in distrs:
        plt.axvline(x=d, linestyle='--', linewidth=1, color='black')
    plt.text(19, .2, 'US House')
    plt.text(60, .2, 'IL Senate')
    plt.text(119, .2, 'IL House')
    plt.title('Illinois')
    plt.savefig(os.path.join(fig_folder, 'il_vary_k.eps'), format='eps', bbox_inches='tight')