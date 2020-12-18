import numpy as np
from scipy.stats import t


def query_tree(leaf_nodes, internal_nodes, query_vals):
    """
    Dynamic programming method to find plan which maximizes linear district metric.
    Args:
        leaf_nodes: (SHPnode list) with node capacity equal to 1 (has no child nodes).
        internal_nodes: (SHPnode list) with node capacity >1 (has child nodes).
        query_vals: (list) of metric values per node.

    Returns: (list, float) tuple of optimal plan and optimal objective value.

    """
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
    """
    Compute the expected seat share as a step function.
    Args:
        district_df: (pd.DataFrame) selected district statistics (requires "mean")
        minimize: (bool) negates values if true

    Returns: (np.array) leaf node query values

    """
    mean = district_df['mean'].values
    return mean < .50 * (-1 if minimize else 1)


def party_advantage_query_fn(district_df, minimize=False):
    """
    Compute the expected seat share as a t distribution.
    Args:
        district_df: (pd.DataFrame) selected district statistics
            (requires "mean", "std_dev", "DoF")
        minimize: (bool) negates values if true

    Returns: (np.array) leaf node query values

    """
    mean = district_df['mean'].values
    std_dev = district_df['std_dev'].values
    DoF = district_df['DoF'].values
    return (1 - t.cdf(.5, DoF, mean, std_dev)) * (-1 if minimize else 1)


def competitive_query_fn(district_df, minimize=False):
    """
    Compute the expected seat swaps.
    Args:
        district_df: (pd.DataFrame) selected district statistics
            (requires "mean", "std_dev", "DoF")
        minimize: (bool) negates values if true

    Returns: (np.array) leaf node query values

    """
    mean = district_df['mean'].values
    std_dev = district_df['std_dev'].values
    DoF = district_df['DoF'].values
    lose_p = t.cdf(.5, DoF, mean, std_dev)
    expected_flips = 2 * (1 - lose_p) * lose_p
    return expected_flips * (-1 if minimize else 1)
