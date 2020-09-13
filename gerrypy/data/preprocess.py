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
from gerrypy.data.data_matching import annotate_precincts
from gerrypy.data.adjacency import create_adjacency_graph
from gerrypy.data.columns import CENSUS_VARIABLE_TO_NAME
from scipy.spatial.distance import pdist, squareform


def load_county_political_data():
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
    """Create adjacency, pairwise dists, construct state_df"""

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


def preprocess_precincts(raw_data_paths,
                       save_path,
                       require_adjacency=False):
    # NOTE: deprecated
    # TODO: update paths with constants
    # TODO: change to "preprocess_precincts" and remove GP stuff
    precinct_shape_path = raw_data_paths['precinct_shape_path']
    census_shape_path = raw_data_paths['census_shape_path']
    census_data_path = raw_data_paths['census_data_path']
    census_column_path = raw_data_paths['census_column_path']

    # Estimate census data for voting precincts
    print('Annotating precincts')
    precinct_df = annotate_precincts(precinct_shape_path, census_shape_path,
                                     census_data_path, census_column_path)

    census_cols = [
        'DP05_0033PE',
        'DP04_0047PE',
        'DP02_0067PE',
        'DP04_0014PE',
        'DP02_0002PE',
        'DP02_0092PE',
        'DP02_0069PE',
        'DP03_0066PE'
    ]

    # Fit a GP to data to learn mean and covariance matrix
    print('Learning mean and covariance matrix')
    features = precinct_df[census_cols]
    y = precinct_df['p_dem'].values
    population = precinct_df['population']
    X = features.values / 100


    # Create state_df and adjacency graph
    precincts = gpd.read_file(precinct_shape_path).to_crs("EPSG:3078")  # m units

    state_df = pd.DataFrame({'x': precincts.centroid.x,
                             'y': precincts.centroid.y})

    if require_adjacency:
        print('Making adjacency graph')
        adjacency_graph = create_adjacency_graph(precincts)
        edge_dists = dict(nx.all_pairs_shortest_path_length(adjacency_graph))

    state_df['population'] = population
    #state_df['affiliation'] = np.clip(mean, 0, 1)

    # Save data necessary to run redistricting
    print('Saving...')
    #np.save(os.path.join(save_path, 'covar'), sigma)
    state_df.to_csv(os.path.join(save_path, 'state_df.csv'), index=False)
    if require_adjacency:
        nx.write_gpickle(adjacency_graph, os.path.join(save_path, 'G.p'))
        json.dump(edge_dists, open(os.path.join(save_path, 'edge_dists.json'), 'w'))


if __name__ == '__main__':
    preprocess_tracts('IL')
    preprocess_tracts('NC')