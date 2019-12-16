from gurobipy import *

def make_kmedians(config, pop_dict, lengths, relax=True):
    n_districts = config['n_districts']
    alpha = config['cost_exponential']
    avg_pop = sum(pop_dict.values()) / n_districts
    pmin = round((1 - config['population_tolerance']) * avg_pop)
    pmax = round((1 + config['population_tolerance']) * avg_pop)

    kmed = Model('Kmedians')
    xs = {i: {} for i in lengths}
    ys = {}
    if relax:
        for i in lengths:
            ys[i] = kmed.addVar(vtype=GRB.CONTINUOUS, lb=0.0,
                                 ub=1.0, name='y%s' % i)
            for j in lengths[i]:
                xs[i][j] = kmed.addVar(vtype=GRB.CONTINUOUS,
                                        lb=0.0, ub=1.0, 
                                        name='x%s,%s' % (i, j))
    else:
        for i in lengths:
            ys[i] = kmed.addVar(vtype=GRB.BINARY, name='y%s' % i)
            for j in lengths[i]:
                xs[i][j] = kmed.addVar(vtype=GRB.BINARY,
                                        name='x%s,%s' % (i, j))

    kmed.addConstr(quicksum(ys[i] for i in lengths) == n_districts,
                   name='n_centers')

    for j in pop_dict:
        kmed.addConstr(quicksum(xs[i][j] for i in lengths if j in xs[i]) == 1,
                       name='one_center_%s' % j)

    kmed.addConstrs(((xs[i][j] <= ys[i]) for i in lengths
                     for j in lengths[i]),
                    name='center_allocation')

    for i in lengths:
        kmed.addConstr(quicksum(pop_dict[j] * xs[i][j]
                                for j in lengths[i]) <= pmax * ys[i],
                       name='population_maximum')
        kmed.addConstr(quicksum(pop_dict[j] * xs[i][j]
                                for j in lengths[i]) >= pmin * ys[i],
                       name='population_minimum')

    kmed.setObjective(quicksum(xs[i][j] * int(lengths[i][j] ** alpha * pop_dict[j] / 1000)
                               for i in lengths
                               for j in lengths[i]),
                      GRB.MINIMIZE)
    kmed.update()

    return kmed, xs, ys