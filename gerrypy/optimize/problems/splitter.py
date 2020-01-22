from gurobipy import *


def make_splitter(config, lengths, population, pop_bounds):
    alpha = config['cost_exponential']
    splitter = Model('dual_split')
    districts = {}
    for center, tracts in lengths.items():
        districts[center] = {}
        for tract in tracts:
            districts[center][tract] = splitter.addVar(vtype=GRB.BINARY,
                                                       name="x1(%s)" % tract)

    for j in population:
        splitter.addConstr(quicksum(districts[i][j] for i in districts
                                    if j in districts[i]) == 1,
                           name='exactlyOne')
    for i in districts:
        splitter.addConstr(quicksum(districts[i][j] * population[j]
                                    for j in districts[i])
                           >= pop_bounds[i]['lb'],
                           name='x%s_minsize' % j)

        splitter.addConstr(quicksum(districts[i][j] * population[j]
                                    for j in districts[i])
                           <= pop_bounds[i]['ub'],
                           name='x%s_maxsize' % j)

    splitter.setObjective(quicksum(districts[i][j] *
                                   int(lengths[i][j] ** alpha * population[j])
                                   for i in lengths
                                   for j in lengths[i]),
                          GRB.MINIMIZE)
    splitter.Params.LogToConsole = 0
    splitter.update()

    return splitter, districts

# def make_splitter_problem(config, lengths, pop_dict):
#     n_districts = config['n_districts']
#     avg_pop = sum(pop_dict.values()) / n_districts
#     pmin = round((1 - config['population_tolerance']) * avg_pop)
#     pmax = round((1 + config['population_tolerance']) * avg_pop)
#     alpha = config['cost_exponential']
#
#     transport = Model('Splitter Problem')
#     transport.Params.TimeLimit = 30
#     xs = {}
#     for i in lengths:
#         xs[i] = {}
#         for j in lengths[i]:
#             xs[i][j] = transport.addVar(vtype=GRB.BINARY,
#                                        name="x%s(%s)" % (i, j))
#
#     for j in pop_dict:
#         transport.addConstr(quicksum(xs[i][j] for i in xs if j in xs[i]) == 1,
#                             name='exactlyOne')
#     for i in xs:
#         transport.addConstr(quicksum(xs[i][j]*pop_dict[j]
#                                     for j in xs[i]) >= pmin,
#                            name='x%s_minsize' % j)
#         transport.addConstr(quicksum(xs[i][j]*pop_dict[j]
#                                     for j in xs[i]) <= pmax,
#                            name='x%s_maxsize' % j)
#
#     transport.setObjective(quicksum(xs[i][j] *
#                             int(lengths[i][j] ** alpha * pop_dict[j])
#                                for i in lengths
#                                for j in lengths[i]),
#                       GRB.MINIMIZE)
#     transport.update()
#
#     return transport, xs
