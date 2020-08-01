import numpy as np
from scipy.stats import t


def query_tree(leaf_nodes, internal_nodes, query_vals):
    nodes = leaf_nodes + internal_nodes
    id_to_ix = {node.id: ix for ix, node in enumerate(leaf_nodes)}
    id_to_node = {node.id: node for node in nodes}
    root = internal_nodes[0] if internal_nodes[0].is_root \
        else [n for n in internal_nodes if n.is_root][0]

    def recursive_query(current_node, all_nodes):
        if not current_node.children_ids:
            return query_vals[id_to_ix[current_node.id]], [current_node.id]

        node_opts = []
        for sample in current_node.children_ids:  # Node partition
            sample_value = 0
            sample_opt_nodes = []
            for child_id in sample:  # partition slice
                child_node = id_to_node[child_id]
                child_value, child_opt = recursive_query(child_node, all_nodes)
                sample_value += child_value
                sample_opt_nodes += child_opt

            node_opts.append((sample_value, sample_opt_nodes))

        return max(node_opts, key=lambda x: x[0])

    return recursive_query(root, nodes)


def party_step_advantage_query_fn(district_df, minimize=False):
    past_elections = district_df[['2008', '2012', '2016']].values
    mean = past_elections.mean(axis=1)
    return mean < .50 * (-1 if minimize else 1)


def party_advantage_query_fn(district_df, minimize=False):
    past_elections = district_df[['2008', '2012', '2016']].values
    mean = past_elections.mean(axis=1)
    std = past_elections.std(axis=1, ddof=1)
    (1 - t.cdf(.5, 2, mean, std)) - mean * (-1 if minimize else 1)


def competitive_query_fn(district_df, minimize=False):
    past_elections = district_df[['2008', '2012', '2016']].values
    mean = past_elections.mean(axis=1)
    std = past_elections.std(axis=1, ddof=1)
    flip_prob = min(1 - t.cdf(.5, 2, mean, std), t.cdf(.5, 2, mean, std))
    return flip_prob.sum() * (-1 if minimize else 1)
