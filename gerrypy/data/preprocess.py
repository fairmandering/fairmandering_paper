import numpy as np
import networkx as nx
import os
import pysal
import json
import pickle
import pandas as pd
import geopandas as gpd
from gerrypy import constants
from gerrypy.data.adjacency import connect_components
from gerrypy.data.load import *
from gerrypy.data.adjacency import create_adjacency_graph
from gerrypy.data.columns import CENSUS_VARIABLE_TO_NAME
from scipy.spatial.distance import pdist, squareform


def load_county_political_data():
    """Helper function to preprocess county voting data to get paritsan vote totals
    for all counties."""
    path = os.path.join(constants.GERRYPY_BASE_PATH, 'data', 'countypres_2000-2016.tab')
    county_results = pd.read_csv(path, sep='\t')
    county_results = county_results.query('party == "democrat" or party == "republican"')
    county_results['FIPS'] = county_results['FIPS'].fillna(0).astype(int).astype(str).apply(lambda x: x.zfill(5))
    calt = county_results.groupby(['year', 'FIPS', 'party']).sum()
    county_year = county_results.groupby(['year', 'FIPS'])
    county_year_votes = county_year['candidatevotes'].sum()
    county_year_vote_p = pd.DataFrame(calt['candidatevotes'] / county_year_votes).query('party == "democrat"')
    county_year_vote_p = county_year_vote_p.reset_index(level=[2], drop=True)
    county_year_vote_p = (1 - county_year_vote_p)
    df = county_year_vote_p.unstack('year')
    county_year_vote_p = df.apply(lambda row: row.fillna(row.mean()), axis=1).stack('year')
    return county_year_vote_p


def preprocess_tracts(state_abbrev):
    """
    Create and save adjacency, pairwise dists, construct state_df
    Args:
        state_abbrev: (str) two letter state abbreviation
    """

    tract_shapes = load_tract_shapes(state_abbrev, constants.ACS_BASE_YEAR)

    state_df = pd.DataFrame({
        'x': tract_shapes.centroid.x,
        'y': tract_shapes.centroid.y,
        'area': tract_shapes.area / 1000**2,  # sq km
        'GEOID': tract_shapes.GEOID.apply(lambda x: str(x).zfill(11)),
    })

    # Join location data with demographic data
    demo_data = pd.read_csv(os.path.join(constants.TRACT_DATA_PATH,
                                         '%d_acs5' % constants.ACS_BASE_YEAR,
                                         '%s_tract.csv' % state_abbrev),
                            low_memory=False)
    demo_data['GEOID'] = demo_data['GEOID'].astype(str).apply(lambda x: x.zfill(11))
    demo_data = demo_data.set_index('GEOID')
    demo_data = demo_data[list(CENSUS_VARIABLE_TO_NAME[str(constants.ACS_BASE_YEAR)])]
    demo_data = demo_data.rename(columns=CENSUS_VARIABLE_TO_NAME[str(constants.ACS_BASE_YEAR)])
    demo_data[demo_data < 0] = 0

    state_df = state_df.set_index('GEOID')
    state_df = state_df.join(demo_data)
    state_df = state_df.reset_index()

    # Load past elections by county and assign uniformly
    county_year_vote_p = load_county_political_data()

    # Missing counties use state average (only 1 in HI)
    ces = set(county_year_vote_p.query('year == 2016').index.get_level_values(0))
    counties = set([g[:5] for g in state_df['GEOID']])
    missing_counties = [c for c in counties if c not in ces]
    presidential_years = [2000, 2004, 2008, 2012, 2016]
    for year in presidential_years:
        for county in missing_counties:
            print('Missing', county, year)
            avg = county_year_vote_p.query('year == @year and FIPS in @counties').mean().item()
            county_year_vote_p.loc[(county, year), 'candidatevotes'] = avg

    for year in presidential_years:
        state_df[str(year)] = state_df.GEOID.apply(lambda x: county_year_vote_p.loc[(x[:5], year)].item())

    shape_list = tract_shapes.geometry.to_list()
    adj_graph = pysal.lib.weights.Rook.from_iterable(shape_list).to_networkx()

    if not nx.is_connected(adj_graph):
        adj_graph = connect_components(tract_shapes)

    edge_dists = dict(nx.all_pairs_shortest_path_length(adj_graph))

    centroids = state_df[['x', 'y']].values
    plengths = squareform(pdist(centroids))

    save_path = os.path.join(constants.OPT_DATA_PATH, state_abbrev)
    os.makedirs(save_path, exist_ok=True)

    state_df.to_csv(os.path.join(save_path, 'state_df.csv'), index=False)
    np.save(os.path.join(save_path, 'lengths.npy'), plengths)
    nx.write_gpickle(adj_graph, os.path.join(save_path, 'G.p'))
    pickle.dump(edge_dists, open(os.path.join(save_path, 'edge_dists.p'), 'wb'))
