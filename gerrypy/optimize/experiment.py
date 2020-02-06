from gerrypy.optimize.cost import expected_rep_gap
from gerrypy.optimize.initial_cols import *
from gerrypy.optimize.problems.master import make_master
from gerrypy.optimize.hierarchical import non_binary_bfs_split
from gerrypy.optimize.tree import SampleTree
from gurobipy import *


def track_tree_sampling(config, state_df, G, lengths, state_covar):
    all_cols = []
    all_costs = []
    maps = []
    trees = []
    obj_vals = []

    # Initialization iteration
    tree = SampleTree(config['hconfig'], config['n_districts'])
    trees.append(tree)
    clean_cols = non_binary_bfs_split(config, G, state_df, lengths, tree)
    all_cols += clean_cols

    costs = [expected_rep_gap(distr,
                              state_df.population.values,
                              state_df.affiliation.values,
                              state_covar) for distr in clean_cols]

    all_costs += costs
    master, variables = make_master(config['n_districts'], len(state_df),
                                    all_cols, all_costs, relax=False)

    master.Params.MIPGap = config['master_obj_tolerance']
    master.update()
    master.optimize()

    acts = [v.X for a, v in variables.items()]
    distrs = [i for i, v in enumerate(acts) if v > .5]
    maps.append({d: all_cols[d] for d in distrs})
    obj_vals.append(master.ObjVal)

    master_constraints = master.getConstrs()

    for i in range(config['n_tree_samples']):
        tree = SampleTree(config['hconfig'], config['n_districts'])
        trees.append(tree)
        sampled_tree = False
        while not sampled_tree:
            try:
                clean_cols = non_binary_bfs_split(config, G, state_df,
                                                  lengths, tree)
                sampled_tree = True
            except ValueError:
                print('Tree sample failed')
                continue
        all_cols += clean_cols
        costs = [expected_rep_gap(distr,
                                  state_df.population.values,
                                  state_df.affiliation.values,
                                  state_covar) for distr in clean_cols]

        all_costs += costs
        for col, cost in zip(clean_cols, costs):
            master_col = Column()
            # Tract membership terms
            master_col.addTerms(np.ones(len(col)),
                                [master_constraints[i] for i in col])
            # n_districts, abs value +, abs value -
            control_terms = [1, cost, cost]
            master_col.addTerms(control_terms, master_constraints[-3:])
            var_num = len(variables)
            variables[var_num] = master.addVar(vtype=GRB.BINARY,
                                               name="x(%s)" % var_num,
                                               column=master_col,
                                               obj=cost)

        master.update()
        master.optimize()

        acts = [v.X for a, v in variables.items()]
        distrs = [i for i, v in enumerate(acts) if v > .5]
        maps.append({d: all_cols[d] for d in distrs})
        obj_vals.append(master.ObjVal)

        print('State Affiliation',
              state_df.affiliation.values.dot(state_df.population.values)
              / state_df.population.values.sum())
        for d in distrs:
            district = all_cols[d]
            pop_array = state_df.population.values[district]
            print('Population:', round(pop_array.sum()),
                  '  | Political Affiliation:',
                  state_df.affiliation.values[district].dot(pop_array)
                  / pop_array.sum())

        if master.ObjVal < config['master_obj_tolerance']:
            break

    return all_cols, all_costs, trees, obj_vals, maps, master
