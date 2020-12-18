from gurobipy import *
import numpy as np
from scipy.stats import t


def make_master(k, block_district_matrix, costs,
                relax=False, opt_type='abs_val'):

    n_blocks, n_columns = block_district_matrix.shape

    master = Model("master LP")

    x = {}
    D = range(n_columns)
    vtype = GRB.CONTINUOUS if relax else GRB.BINARY
    for j in D:
        x[j] = master.addVar(vtype=vtype, name="x(%s)" % j)

    master.addConstrs((quicksum(x[j] * block_district_matrix[i, j] for j in D) == 1
                       for i in range(n_blocks)), name='exactlyOne')

    master.addConstr(quicksum(x[j] for j in D) == k,
                     name="totalDistricts")

    if opt_type == 'minimize':
        master.setObjective(quicksum(costs[j] * x[j] for j in D), GRB.MINIMIZE)
    elif opt_type == 'maximize':
        master.setObjective(quicksum(costs[j] * x[j] for j in D), GRB.MAXIMIZE)
    elif opt_type == 'abs_val':
        w = master.addVar(name="w", lb=-k, ub=k)
        master.addConstr(quicksum(costs[j] * x[j] for j in D) <= w,
                         name='absval_pos')
        master.addConstr(quicksum(costs[j] * x[j] for j in D) >= -w,
                         name='absval_neg')

        master.setObjective(w, GRB.MINIMIZE)
    else:
        raise ValueError('Invalid optimization type')

    return master, x


def efficiency_gap_coefficients(district_df, state_vote_share):
    mean = district_df['mean'].values
    std_dev = district_df['std_dev'].values
    DoF = district_df['DoF'].values
    expected_seats = 1 - t.cdf(.5, DoF, mean, std_dev)
    # https://www.brennancenter.org/sites/default/files/legal-work/How_the_Efficiency_Gap_Standard_Works.pdf
    # Efficiency Gap = (Seat Margin – 50%) – 2 (Vote Margin – 50%)
    return (expected_seats - .5) - 2 * (state_vote_share - .5)


def make_root_partition_to_leaf_map(leaf_nodes, internal_nodes):
    def add_children(node, root_partition_id):
        if node.n_districts > 1:
            for partition in node.children_ids:
                for child in partition:
                    add_children(node_dict[child], root_partition_id)
        else:
            node_to_root_partition[id_to_ix[node.id]] = root_partition_id

    # Create mapping from leaf ix to root partition ix
    node_to_root_partition = {}
    node_dict = {n.id: n for n in internal_nodes + leaf_nodes}
    id_to_ix = {n.id: ix for ix, n in enumerate(leaf_nodes)}
    root = internal_nodes[0]
    for ix, root_partition in enumerate(root.children_ids):
        for child in root_partition:
            add_children(node_dict[child], ix)

    # Create inverse mapping
    partition_map = {}
    for node_ix, partition_ix in node_to_root_partition.items():
        try:
            partition_map[partition_ix].append(node_ix)
        except KeyError:
            partition_map[partition_ix] = [node_ix]
    partition_map = {ix: np.array(leaf_list) for ix, leaf_list in partition_map.items()}

    return partition_map
