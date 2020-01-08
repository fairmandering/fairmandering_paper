from gurobipy import *


def make_transportation_problem(config, lengths, pop_dict):
    n_districts = config['n_districts']
    avg_pop = sum(pop_dict.values()) / n_districts
    pmin = round((1 - config['population_tolerance']) * avg_pop)
    pmax = round((1 + config['population_tolerance']) * avg_pop)
    alpha = config['cost_exponential']

    transport = Model('Transportation Problem')
    transport.Params.TimeLimit = 30
    xs = {}
    for i in lengths:
        xs[i] = {}
        for j in lengths[i]:
            xs[i][j] = transport.addVar(vtype=GRB.BINARY,
                                       name="x%s(%s)" % (i, j))

    for j in pop_dict:
        transport.addConstr(quicksum(xs[i][j] for i in xs if j in xs[i]) == 1,
                            name='exactlyOne')
    for i in xs:
        transport.addConstr(quicksum(xs[i][j]*pop_dict[j]
                                    for j in xs[i]) >= pmin,
                           name='x%s_minsize' % j)
        transport.addConstr(quicksum(xs[i][j]*pop_dict[j]
                                    for j in xs[i]) <= pmax,
                           name='x%s_maxsize' % j)

    transport.setObjective(quicksum(xs[i][j] *
                            int(lengths[i][j] ** alpha * pop_dict[j])
                               for i in lengths
                               for j in lengths[i]),
                      GRB.MINIMIZE)
    transport.update()

    return transport, xs