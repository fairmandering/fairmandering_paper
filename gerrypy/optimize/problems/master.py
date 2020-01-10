from gurobipy import *
import numpy as np

def make_master(config, initial_cols, costs, relax=True):
    n_districts = config['n_districts']
    n_blocks = config['synmap_config']['width'] * \
               config['synmap_config']['height']

    district_mat = np.zeros((n_blocks, len(costs)))
    for ix, column in enumerate(initial_cols):
        district_mat[column, ix] = 1

    master = Model("master LP")
    x = {}
    J = range(len(initial_cols))

    vtype = GRB.CONTINUOUS if relax else GRB.BINARY
    for j in J:
        x[j] = master.addVar(vtype=vtype, name="x(%s)" % j)

    w = master.addVar(name="w")

    master.addConstr(quicksum(costs[j] * x[j] for j in J) <= w,
                     name='absval_pos')
    master.addConstr(quicksum(costs[j] * x[j] for j in J) >= -w,
                     name='absval_neg')

    master.addConstrs((quicksum(x[j] * district_mat[i, j] for j in J) == 1
                       for i in range(n_blocks)), name='exactlyOne')

    master.addConstr(quicksum(x[j] for j in J) == n_districts,
                     name="totalDistricts")

    master.setObjective(w, GRB.MINIMIZE)

    return master, x


