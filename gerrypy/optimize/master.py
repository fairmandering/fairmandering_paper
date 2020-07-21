from gurobipy import *
import numpy as np


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


