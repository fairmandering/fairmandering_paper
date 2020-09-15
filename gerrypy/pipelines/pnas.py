import os
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gerrypy import constants
from gerrypy.analyze import tree
from gerrypy.analyze import subsample
from gerrypy.analyze import districts
from gerrypy.data.load import *


def run_all_states_result_pipeline(result_path, test=False):
    if test:
        states = ['AL', 'CO', 'WV']
    else:
        states = [state for state in constants.seats
                  if constants.seats[state]['house'] > 1]

    pipeline_result = {}
    for state in states:
        state_results = {}
        tree_path = glob.glob(os.path.join(result_path, '%s_*.p' % state))[0]
        tree_data = pickle.load(open(tree_path, 'rb'))
        district_df = pd.read_csv(os.path.join(result_path, '%s.csv', 'rb'))

        leaf_nodes = tree_data['leaf_nodes']
        internal_nodes = tree_data['internal_nodes']

        if 'dispersion' not in set(district_df.columns):
            state_df = load_state_df(state)
            district_list = [d.area for d in internal_nodes]
            district_df['dispersion'] = districts.dispersion_compactness(district_list, state_df)

        extreme_data = extreme_solutions(leaf_nodes, internal_nodes)
        

def extreme_solutions(leaf_nodes, internal_nodes, district_df):
    extreme_data = {}
    r_advantage_query_vals = tree.party_advantage_query_fn(district_df)
    d_advantage_query_vals = 1 - r_advantage_query_vals
    r_val, r_sol = tree.query_tree(leaf_nodes, internal_nodes, r_advantage_query_vals)
    d_val, d_sol = tree.query_tree(leaf_nodes, internal_nodes, d_advantage_query_vals)
    extreme_data['r_advantage'] = {
        'objective_value': r_val,
        'solution': {n.id: n.area for n in leaf_nodes if n.id in set(r_sol)}
    }
    extreme_data['d_advantage'] = {
        'objective_value': d_val,
        'solution': {n.id: n.area for n in leaf_nodes if n.id in set(d_sol)}
    }

    uncompact_query_vals = district_df.dispersion.values
    compact_query_vals = 1 - uncompact_query_vals
    uncompact_val, uncompact_sol = tree.query_tree(leaf_nodes, internal_nodes, uncompact_query_vals)
    compact_val, compact_sol = tree.query_tree(leaf_nodes, internal_nodes, compact_query_vals)
    extreme_data['uncompact'] = {
        'objective_value': uncompact_val,
        'solution': {n.id: n.area for n in leaf_nodes if n.id in set(uncompact_sol)}
    }
    extreme_data['compact'] = {
        'objective_value': compact_val,
        'solution': {n.id: n.area for n in leaf_nodes if n.id in set(compact_sol)}
    }

    competitive_query_vals = tree.competitive_query_fn(district_df)
    uncompetitive_query_vals = -competitive_query_vals
    uncompetitive_val, uncompetitive_sol = tree.query_tree(leaf_nodes, internal_nodes, competitive_query_vals)
    competitive_val, competitive_sol = tree.query_tree(leaf_nodes, internal_nodes, uncompetitive_query_vals)
    extreme_data['uncompetitive'] = {
        'objective_value': -uncompetitive_val,
        'solution': {n.id: n.area for n in leaf_nodes if n.id in set(uncompetitive_sol)}
    }
    extreme_data['competitive'] = {
        'objective_value': competitive_val,
        'solution': {n.id: n.area for n in leaf_nodes if n.id in set(competitive_sol)}
    }

    return extreme_data

