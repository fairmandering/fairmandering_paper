import pandas as pd
import numpy as np
from scipy.stats import t
import math
import itertools
from scipy.spatial.distance import pdist, cdist
from gerrypy.data.load import *
from gerrypy.analyze.tree import *


def average_entropy(M):
    return (- M * np.ma.log(M).filled(0) - (1 - M) *
            np.ma.log(1 - M).filled(0)).sum() / (M.shape[0] * M.shape[1])


def svd_entropy(sigma):
    sigma_hat = sigma / sigma.sum()
    entropy = - (sigma_hat * (np.ma.log(sigma_hat).filled(0) / np.log(math.e))).sum()
    return entropy / (math.log(len(sigma)) / math.log(math.e))


def make_bdm(leaf_nodes, n_blocks=None):
    districts = [d.area for d in leaf_nodes]
    if n_blocks is None:
        n_blocks = max([max(d) for d in districts]) + 1
    block_district_matrix = np.zeros((n_blocks, len(districts)))
    for ix, d in enumerate(districts):
        block_district_matrix[d, ix] = 1
    return block_district_matrix


def bdm_metrics(block_district_matrix, k):
    ubdm = np.unique(block_district_matrix, axis=1)
    max_rank = min(ubdm.shape)
    U, Sigma, Vt = np.linalg.svd(ubdm)

    Dsim = 1 - pdist(ubdm.T, metric='jaccard')

    precinct_coocc = ubdm @ ubdm.T
    precinct_conditional_p = precinct_coocc / ubdm.sum(axis=1)

    L = (np.diag(precinct_coocc.sum(axis=1)) - precinct_coocc)
    D_inv = np.diag(precinct_coocc.sum(axis=1) ** -.5)
    e = np.linalg.eigvals(D_inv @ L @ D_inv)

    conditional_entropy = average_entropy(precinct_conditional_p)

    return {
        'conditional_entropy': conditional_entropy,
        'average_district_sim': k * np.average(Dsim),
        'svd_entropy': svd_entropy(Sigma),
        '50p_approx_rank': sum(np.cumsum(Sigma / Sigma.sum()) < .5) / max_rank,
        '95p_approx_rank': sum(np.cumsum(Sigma / Sigma.sum()) < .95) / max_rank,
        '99p_approx_rank': sum(np.cumsum(Sigma / Sigma.sum()) < .99) / max_rank,
        'lambda_2': e[1],
        'lambda_k': e[k],
    }


def generation_metrics(cg, low_memory=False):
    p_infeasible = cg.n_infeasible_partitions / \
                   (cg.n_infeasible_partitions + cg.n_successful_partitions)
    n_interior_nodes = len(cg.internal_nodes)
    districts = [d.area for d in cg.leaf_nodes]
    duplicates = len(districts) - len(set([frozenset(d) for d in districts]))

    # TODO: deduplicate compactness computation
    dispersion = np.array(dispersion_compactness(districts, cg.state_df))
    roeck = np.array(roeck_compactness(districts, cg.state_df, cg.lengths))
    cut_edges = np.array(list(map(lambda x: sum(1 for _ in nx.edge_boundary(cg.G, x)), districts)))
    min_compactness, _ = query_tree(cg.leaf_nodes, cg.internal_nodes, cut_edges)
    max_compactness, _ = query_tree(cg.leaf_nodes, cg.internal_nodes, -cut_edges)
    compactness_disparity = - min_compactness / max_compactness

    block_district_matrix = make_bdm(cg.leaf_nodes)
    district_df = create_district_df(cg.config['state'], block_district_matrix)

    expected_seats = party_advantage_query_fn(district_df)
    max_seats, _ = query_tree(cg.leaf_nodes, cg.internal_nodes, expected_seats)
    min_seats, _ = query_tree(cg.leaf_nodes, cg.internal_nodes, -expected_seats)
    seat_disparity = max_seats + min_seats  # Min seats is negative

    metrics = {
        'n_root_failures': cg.failed_root_samples,
        'p_infeasible': p_infeasible,
        'n_interior_nodes': n_interior_nodes,
        'n_districts': len(districts),
        'p_duplicates': duplicates / len(districts),
        'dispersion': np.array(dispersion).mean(),
        'roeck': np.array(roeck).mean(),
        'cut_edges': np.array(cut_edges).mean(),
        'compactness_disparity': compactness_disparity,
        'seat_disparity': seat_disparity
    }
    if low_memory:
        return metrics
    else:
        return {**metrics, **bdm_metrics(block_district_matrix, cg.config['n_districts'])}


def number_of_districtings(leaf_nodes, interior_nodes):
    nodes = leaf_nodes + interior_nodes
    id_to_node = {node.id: node for node in nodes}
    root = interior_nodes[0] if interior_nodes[0].is_root \
        else [n for n in interior_nodes if n.is_root][0]

    def recursive_compute(current_node, all_nodes):
        if not current_node.children_ids:
            return 1

        total_districtings = 0
        for sample in current_node.children_ids:
            sample_districtings = 1
            for child_id in sample:
                child_node = id_to_node[child_id]
                sample_districtings *= recursive_compute(child_node, all_nodes)

            total_districtings += sample_districtings
        return total_districtings

    return recursive_compute(root, nodes)


def enumerate_partitions(leaf_nodes, interior_nodes):
    def feasible_partitions(node, node_dict):
        if not node.children_ids:
            return [[node.id]]

        partitions = []
        for disjoint_sibling_set in node.children_ids:
            sibling_partitions = []
            for child in disjoint_sibling_set:
                sibling_partitions.append(feasible_partitions(node_dict[child],
                                                              node_dict))
            combinations = [list(itertools.chain.from_iterable(combo))
                            for combo in itertools.product(*sibling_partitions)]
            partitions.append(combinations)

        return list(itertools.chain.from_iterable(partitions))

    root = interior_nodes[0] if interior_nodes[0].is_root \
        else [n for n in interior_nodes if n.is_root][0]

    node_dict = {n.id: n for n in interior_nodes + leaf_nodes}
    return feasible_partitions(root, node_dict)


def enumerate_distribution(leaf_nodes, interior_nodes, leaf_values):
    def feasible_partitions(node, node_dict):
        if not node.children_ids:
            return [[leaf_dict[node.id]]]

        partitions = []
        for disjoint_sibling_set in node.children_ids:
            sibling_partitions = []
            for child in disjoint_sibling_set:
                sibling_partitions.append(feasible_partitions(node_dict[child],
                                                              node_dict))
            combinations = [list(itertools.chain.from_iterable(combo))
                            for combo in itertools.product(*sibling_partitions)]
            partitions.append(combinations)
        return [[sum(c)] for c in list(itertools.chain.from_iterable(partitions))]

    root = interior_nodes[0] if interior_nodes[0].is_root \
        else [n for n in interior_nodes if n.is_root][0]

    leaf_dict = {n.id: leaf_values[ix] for ix, n in enumerate(leaf_nodes)}
    node_dict = {n.id: n for n in interior_nodes + leaf_nodes}
    plan_values = feasible_partitions(root, node_dict)
    return [item for sublist in plan_values for item in sublist]


def roeck_compactness(districts, state_df, lengths):
    compactness_scores = []
    for d in districts:
        area = state_df.loc[d]['area'].sum()
        radius = lengths[np.ix_(d, d)].max() / 2
        circle_area = radius ** 2 * math.pi
        roeck = area / circle_area
        compactness_scores.append(roeck)
    return compactness_scores


def roeck_more_exact(districts, state_df, tracts, lengths):
    def unwind_coords(poly):
        try:
            return np.array(poly.exterior.coords)
        except AttributeError:
            return np.concatenate([p.exterior.coords for p in poly])
    compactness_scores = []
    for d in districts:
        area = state_df.loc[d]['area'].sum()
        pairwise_dists = lengths[np.ix_(d, d)]
        max_pts = np.unravel_index(np.argmax(pairwise_dists), pairwise_dists.shape)
        t1, t2 = max_pts
        p1 = unwind_coords(tracts.loc[d[t1]].geometry)
        p2 = unwind_coords(tracts.loc[d[t2]].geometry)
        radius = np.max(cdist(p1, p2)) / 2000
        circle_area = radius ** 2 * math.pi
        roeck = area / circle_area
        compactness_scores.append(roeck)
    return compactness_scores


def dispersion_compactness(districts, state_df):
    compactness_scores = []
    for d in districts:
        population = state_df.loc[d]['population'].values
        dlocs = state_df.loc[d][['x', 'y']].values
        centroid = np.average(dlocs, weights=population, axis=0)
        geo_dispersion = (np.subtract(dlocs, centroid) ** 2).sum(axis=1) ** .5 / 1000
        dispersion = np.average(geo_dispersion, weights=population)
        compactness_scores.append(dispersion)
    return compactness_scores


def create_district_df(state, bdm, calculate_compactness=True):
    election_df = load_election_df(state)
    state_df = load_state_df(state)
    district_list = []
    if election_df is not None:
        vote_total_columns = list(election_df.columns)
        state_df = pd.concat([state_df, election_df], axis=1)
    else:
        vote_total_columns = []
    sum_columns = ['area', 'population'] + vote_total_columns
    bdm = bdm.astype(bool)
    for row in bdm.T:
        district_tract_df = state_df.iloc[row]
        distr_series = pd.Series(np.average(district_tract_df,
                                            weights=district_tract_df['population'], axis=0),
                                 index=state_df.columns)
        distr_series[sum_columns] = district_tract_df[sum_columns].sum(axis=0)
        district_list.append(distr_series)

    district_df = pd.DataFrame(district_list)
    if vote_total_columns:
        elections = list(set([e[2:] for e in vote_total_columns]))
        share_df = pd.DataFrame({
            e: district_df['R_' + e] / (district_df['R_' + e] + district_df['D_' + e])
            for e in elections
        })
    else:  # If no precinct election data use county
        elections = ['2008', '2012', '2016']
        share_df = district_df[elections]

    district_df['mean'] = share_df.mean(axis=1)
    district_df['std_dev'] = share_df.std(ddof=1, axis=1)
    district_df['DoF'] = len(elections) - 1

    if calculate_compactness:
        state_df, G, lengths, _ = load_opt_data(state)
        districts = [list(np.nonzero(row)[0]) for row in bdm.T]

        dispersion = dispersion_compactness(districts, state_df)
        roeck = roeck_compactness(districts, state_df, lengths)
        cut_edges = list(map(lambda x: sum(1 for _ in nx.edge_boundary(G, x)), districts))

        district_df['dispersion'] = dispersion
        district_df['roeck'] = roeck
        district_df['cut_edges'] = cut_edges

    return district_df
