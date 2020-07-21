from gurobipy import *


def make_partition_IP(lengths, edge_dists, G, population, pop_bounds, alpha):
    splitter = Model('partition')
    districts = {}
    for center, tracts in lengths.items():
        districts[center] = {}
        for tract in tracts:
            districts[center][tract] = splitter.addVar(vtype=GRB.BINARY)

    for j in population:
        splitter.addConstr(quicksum(districts[i][j] for i in districts
                                    if j in districts[i]) == 1,
                           name='exactlyOne')
    for i in districts:
        splitter.addConstr(quicksum(districts[i][j] * population[j]
                                    for j in districts[i])
                           >= pop_bounds[i]['lb'],
                           name='x%s_minsize' % i)

        splitter.addConstr(quicksum(districts[i][j] * population[j]
                                    for j in districts[i])
                           <= pop_bounds[i]['ub'],
                           name='x%s_maxsize' % i)

    # Connectivity
    connectivity_constr = {}
    for center in districts:
        connectivity_constr[center] = {}
        for node in districts[center]:
            constr_set = []
            dist = edge_dists[center][node]
            for nbor in G[node]:
                if edge_dists[center][nbor] == dist - 1\
                        and nbor in districts[center]:
                    constr_set.append(nbor)
            connectivity_constr[center][node] = constr_set

    for center, sp_sets in connectivity_constr.items():
        for node, sp_set in sp_sets.items():
            if center == node:
                continue
            splitter.addConstr(districts[center][node] <=
                               quicksum(districts[center][nbor]
                                        for nbor in sp_set))

    splitter.setObjective(quicksum(districts[i][j] *
                                   int(lengths[i][j] ** alpha * population[j])
                                   for i in lengths
                                   for j in lengths[i]),
                          GRB.MINIMIZE)
    splitter.Params.LogToConsole = 0
    splitter.Params.TimeLimit = len(population) / 200
    splitter.update()

    return splitter, districts
