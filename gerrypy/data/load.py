import pickle
from gerrypy import constants
import networkx as nx
import os
import numpy as np
import pandas as pd
import geopandas as gpd


def load_state_df(state_abbrev):
    state_df_path = os.path.join(constants.OPT_DATA_PATH,
                                 state_abbrev,
                                 'state_df.csv')
    df = pd.read_csv(state_df_path)
    return df.sort_values(by='GEOID').reset_index(drop=True)


def load_election_df(state_abbrev):
    election_df_path = os.path.join(constants.OPT_DATA_PATH,
                                    state_abbrev,
                                    'election_df.csv')
    try:
        df = pd.read_csv(election_df_path)
    except FileNotFoundError:
        df = None
    return df  # Indices are equal to state_df integer indices


def load_acs(state_abbrev, year=None, county=False):
    base_path = constants.COUNTY_DATA_PATH if county else constants.TRACT_DATA_PATH
    name_extension = 'county' if county else 'tract'
    year = year if year else constants.ACS_BASE_YEAR
    state_path = os.path.join(base_path,
                              '%s_acs5' % str(year),
                              '%s_%s.csv' % (state_abbrev, name_extension))
    return pd.read_csv(state_path, low_memory=False).sort_values('GEOID').reset_index(drop=True)


def load_tract_shapes(state_abbrev, year=None):
    if not year:
        year = constants.ACS_BASE_YEAR
    shape_fname = state_abbrev + '_' + str(year)
    tract_shapes = gpd.read_file(os.path.join(constants.CENSUS_SHAPE_PATH,
                                              shape_fname))
    tract_shapes = tract_shapes.to_crs("EPSG:3078")  # meters
    tract_shapes = tract_shapes[tract_shapes.ALAND > 0]
    return tract_shapes.sort_values(by='GEOID').reset_index(drop=True)


def load_district_shapes(state=None, year=2018):
    path = os.path.join(constants.GERRYPY_BASE_PATH, 'data',
                        'district_shapes', 'cd_' + str(year))
    gdf = gpd.read_file(path).sort_values('GEOID').to_crs("EPSG:3078")  # meters
    if state is not None:
        state_geoid = str(constants.ABBREV_DICT[state][constants.FIPS_IX])
        return gdf[gdf.STATEFP == state_geoid]
    else:
        return gdf


def load_opt_data(state_abbrev):
    data_base_path = os.path.join(constants.OPT_DATA_PATH, state_abbrev)
    state_df_path = os.path.join(data_base_path, 'state_df.csv')
    adjacency_graph_path = os.path.join(data_base_path, 'G.p')

    state_df = pd.read_csv(state_df_path)
    G = nx.read_gpickle(adjacency_graph_path)

    if os.path.exists(os.path.join(data_base_path, 'lengths.npy')):
        lengths_path = os.path.join(data_base_path, 'lengths.npy')
        lengths = np.load(lengths_path)
    else:
        from scipy.spatial.distance import pdist, squareform
        lengths = squareform(pdist(state_df[['x', 'y']].values))

    if os.path.exists(os.path.join(data_base_path, 'edge_dists.p')):
        edge_dists_path = os.path.join(data_base_path, 'edge_dists.p')
        edge_dists = pickle.load(open(edge_dists_path, 'rb'))
    else:
        edge_dists = dict(nx.all_pairs_shortest_path_length(G))

    return state_df, G, lengths, edge_dists
