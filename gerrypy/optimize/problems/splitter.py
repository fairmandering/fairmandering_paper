from gurobipy import *


def make_splitter(lengths, population, pop_bounds, alpha):
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
