import os
import pandas as pd
from gerrypy import constants
from gerrypy.paper.acda.all_states import *


def load_results():
    recom_dir = os.path.join(constants.RESULTS_PATH, 'recombination_benchmark')
    results = {}
    for file in os.listdir(recom_dir):
        state = file[:2]
        k = constants.seats[state]['house']
        result_df = pd.read_csv(os.path.join(recom_dir, file))
        result_df['cut_edges'] /= (k / 2)
        result_df['expected_R_seats'] /= k
        results[file[:2]] = result_df.describe().iloc[3:]

    combined_df = pd.concat(results.values(), keys=results.keys())
    cut_edges_df = combined_df['cut_edges'].unstack(level=0)
    ess_df = combined_df['expected_R_seats'].unstack(level=0)

    ensemble_dir = os.path.join(constants.RESULTS_PATH,
                                "allstates", "aaai_columns1595813019", "pnas_results")
    ensemble_results = load_all_state_results(ensemble_dir)
    shp_ess_df = create_seat_share_box_df(ensemble_results, np.random.random(len(ensemble_results)))
    shp_cut_edges_df = create_compactness_box_df(ensemble_results, 'cut_edges', np.random.random(len(ensemble_results)))
    shp_ess_df = shp_ess_df[ess_df.columns]
    shp_cut_edges_df = shp_cut_edges_df[cut_edges_df.columns]

    combined_ess = pd.concat([shp_ess_df, ess_df], keys=['SHP', 'recom'])
    combined_ess = pd.DataFrame(combined_ess.stack()).reset_index()
    name_map = {old: new for old, new in zip(combined_ess.columns, ['method', 'percentile', 'state', 'value'])}
    combined_ess = combined_ess.rename(columns=name_map)

    combined_cut_edges = pd.concat([shp_cut_edges_df, cut_edges_df], keys=['SHP', 'recom'])
    combined_cut_edges = pd.DataFrame(combined_cut_edges.stack()).reset_index()
    name_map = {old: new for old, new in zip(combined_cut_edges.columns, ['method', 'percentile', 'state', 'value'])}
    combined_cut_edges = combined_cut_edges.rename(columns=name_map)

    return combined_ess, combined_cut_edges