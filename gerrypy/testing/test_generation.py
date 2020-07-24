import copy
from gerrypy.optimize.generate import ColumnGenerator
from gerrypy.analyze.districts import *

center_selection_config = {
    'selection_method': 'uncapacitated_kmeans',  # one of
    'perturbation_scale': 1,
    'n_random_seeds': 0,
    'capacities': 'match',
    'capacity_weights': 'voronoi'
}
tree_config = {
    'max_sample_tries': 100,
    'n_samples': 1,
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
    'IP_gap_tol': 1e-4,
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


def test_init():
    cg = ColumnGenerator(base_config)
    assert cg is not None
    assert cg.config['ideal_pop'] > 100
    assert 0 < cg.config['max_pop_variation'] < 1000000
    assert 100 < cg.state_df['population'].sum() < 1e10
    assert len(cg.state_df) > 0
    assert len(cg.state_df) == len(cg.G) == len(cg.lengths) == len(cg.edge_dists)
    assert cg.root is None
    print('test_init success')


def test_gen():
    cg = ColumnGenerator(base_config)
    cg.generate()
    assert len(cg.leaf_nodes) == base_config['n_districts']
    assert cg.sample_queue == []
    assert 0 < len(cg.internal_nodes) <= len(cg.leaf_nodes)

    print('test_gen success')


def test_metrics():
    config = copy.deepcopy(base_config)
    config['n_samples'] = 3
    config['n_root_samples'] = 3
    cg = ColumnGenerator(config)
    cg.generate()
    bdm = make_bdm(cg.leaf_nodes)
    assert bdm.shape == (len(cg.state_df), len(cg.leaf_nodes))
    metrics = generation_metrics(cg)
    assert 0 <= metrics['p_infeasible'] < 1
    assert 0 <= metrics['p_duplicates'] < 1
    assert 0 <= metrics['conditional_entropy'] <= 1
    assert 0 <= metrics['svd_entropy'] <= 1
    assert 0 <= metrics['50p_approx_rank'] <= 1
    assert 0 <= metrics['95p_approx_rank'] <= 1
    assert 0 <= metrics['99p_approx_rank'] <= 1
    assert 0 <= metrics['lambda_2'] <= 1
    assert 0 <= metrics['lambda_k'] <= 1
    n_plans = number_of_districtings(cg.leaf_nodes, cg.internal_nodes)
    plans = enumerate_partitions(cg.leaf_nodes, cg.internal_nodes)
    assert len(plans) == n_plans
    ddf = create_district_df(bdm, cg.state_df)
    assert len(ddf) == len(cg.leaf_nodes)
    pop_error = np.abs(ddf['population'] - cg.config['ideal_pop'])
    assert np.all(pop_error < cg.config['max_pop_variation'])
    assert len(cg.failed_regions) == cg.failed_root_samples
    print('test_metrics success')


if __name__ == '__main__':
    test_init()
    test_gen()
    test_metrics()
