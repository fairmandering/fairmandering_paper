import gerrypy.constants
from gerrypy.data.load import *
from gerrypy.analyze.viz import *
from gerrypy.analyze.tree import *
from gerrypy.data.precinct_state_wrappers import wrappers

from scipy.stats import t
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time

from gerrychain import (GeographicPartition, MarkovChain,
                        updaters, constraints, accept, Election)
from gerrychain.updaters import cut_edges
from gerrychain.proposals import recom
from gerrychain.tree import recursive_tree_part
from functools import partial


def load_initial_plan(ensemble_path, ensemble_name, how):
    ensemble = pickle.load(open(os.path.join(ensemble_path, ensemble_name + '.p'),
                                'rb'))
    ddf = pd.read_csv(os.path.join(ensemble_path, 'district_dfs',
                                   ensemble_name + '_district_df.csv'))
    leaf_nodes = ensemble['leaf_nodes']
    internal_nodes = ensemble['internal_nodes']
    if how == 'random':
        query_vals = np.random.random(len(ddf))
        obj, node_ids = query_tree(leaf_nodes, internal_nodes, query_vals)
    elif how == 'max_edge_cuts':
        query_vals = ddf['cut_edges']
        obj, node_ids = query_tree(leaf_nodes, internal_nodes, query_vals)
        print(obj)
    else:
        raise RuntimeError('No such way to load plan')
    id_to_area = {n.id: n.area for n in leaf_nodes if n.id in set(node_ids)}
    plan = {int(i+1): area for i, area in enumerate(id_to_area.values())}
    return plan, len(ddf)


def convert_opt_data_to_gerrychain_input(state, opt_data='', plan=None):
    state_df, adj_graph, _, _ = load_opt_data(state, opt_data)
    election_df = load_election_df(state, opt_data)

    state_df = state_df[['population']]
    metric_df = pd.concat([state_df, election_df], axis=1)

    if plan is not None:
        plan_inverse_map = {}
        for district_ix, tract_ixs in plan.items():
            for tix in tract_ixs:
                plan_inverse_map[tix] = district_ix
        metric_df['initial_plan'] = pd.Series(plan_inverse_map, dtype=str)
    else:
        k = constants.seats[state]['house']
        ideal_pop = state_df.population.sum() / k
        nx.set_node_attributes(adj_graph, metric_df.T.to_dict())
        plan = recursive_tree_part(adj_graph, range(k), ideal_pop, 'population', 0.01)
        metric_df['initial_plan'] = pd.Series(plan, dtype=str)

    nx.set_node_attributes(adj_graph, metric_df.T.to_dict())

    elections = [
        Election(election, {'Democratic': 'D_' + election, 'Republican': 'R_' + election})
        for election in wrappers[state]().election_columns(include_party=False)
    ]
    return adj_graph, elections


def expected_seat_share_updater(partition):
    elecs = partition.elections
    election_array = np.array([partition[e].percents("Republican")
                               for e in elecs]).T
    mean_vote_share = election_array.mean(axis=1)
    vote_share_std = election_array.std(axis=1, ddof=1)
    degrees_of_freedom = len(elecs)
    return (1 - t.cdf(.5, degrees_of_freedom, mean_vote_share, vote_share_std)).sum()


def run_chain(adj_graph, elections, total_steps=100):
    my_updaters = {
        "population": updaters.Tally("population"),
        "cut_edges": cut_edges,
        "elections": lambda x: [elec.alias for elec in elections],
        "expected_seat_share": expected_seat_share_updater
    }
    election_updaters = {election.name: election for election in elections}
    my_updaters.update(election_updaters)

    initial_partition = GeographicPartition(adj_graph, assignment="initial_plan",
                                            updaters=my_updaters)
    ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)
    proposal = partial(recom,
                       pop_col="population",
                       pop_target=ideal_population,
                       epsilon=0.01,
                       node_repeats=2
                       )
    pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.01)
    chain = MarkovChain(
        proposal=proposal,
        constraints=[
            pop_constraint,
        ],
        accept=accept.always_accept,
        initial_state=initial_partition,
        total_steps=total_steps
    )

    return pd.DataFrame({ix: {
        "cut_edges": len(partition["cut_edges"]),
        "expected_R_seats": partition["expected_seat_share"]}
        for ix, partition in enumerate(chain)
    }).T


if __name__ == "__main__":
    TEST = True
    ENSEMBLE_PATH = os.path.join(constants.RESULTS_PATH, 'allstates',
                                 'aaai_columns1595813019')
    EXPERIMENT_NAME = "recombination_municipal_boundary_test"
    CUSTOM_INPUT = "municipal_hard_constraint"
    STATES = ["PA", "TX"]

    ensembles = {file[:2]: file[:-2] for file in os.listdir(ENSEMBLE_PATH)
                 if file[-2:] == '.p' and file[:2] in set(STATES)}
    os.makedirs(os.path.join(constants.RESULTS_PATH, EXPERIMENT_NAME), exist_ok=True)
    for state, file in ensembles.items():
        if not CUSTOM_INPUT:
            plan, n_generated_districts = load_initial_plan(
                ENSEMBLE_PATH, file, how='max_edge_cuts')
        adj_graph, elections = convert_opt_data_to_gerrychain_input(
            state, opt_data=CUSTOM_INPUT)
        warmup_steps = constants.seats[state]['house'] * 10
        if TEST:
            total_steps = 10
        else:
            total_steps = warmup_steps + n_generated_districts / 2
        print('Running ReCom for state %s with %d steps...' % (state, total_steps))

        start_t = time.time()
        result_df = run_chain(adj_graph, elections, total_steps)
        end_t = time.time()

        runtime = end_t - start_t
        print(runtime)
        save_name = os.path.join(constants.RESULTS_PATH, EXPERIMENT_NAME,
                                 ''.join([state, str(round(runtime)), '.csv']))
        result_df.to_csv(save_name, index=False)