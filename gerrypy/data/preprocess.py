import numpy as np
import networkx as nx
import os
import pysal
import json
import pickle
import pandas as pd
import geopandas as gpd
from gerrypy import constants
from gerrypy.data.data_matching import annotate_precincts
from gerrypy.data.adjacency import create_adjacency_graph
from scipy.spatial.distance import pdist, squareform


def preprocess_tracts(state_abbrev):
    """Create adjacency, pairwise dists, construct state_df"""
    state_fips = constants.ABBREV_DICT[state_abbrev][constants.FIPS_IX]
    shape_fname = state_fips + '_' + str(constants.ACS_BASE_YEAR)
    tract_shapes = gpd.read_file(os.path.join(constants.CENSUS_SHAPE_PATH, shape_fname))
    tract_shapes = tract_shapes.to_crs("EPSG:3078")  # meters
    tract_shapes = tract_shapes[tract_shapes.ALAND > 0]  # IL tracts

    state_df = pd.DataFrame({
        'x': tract_shapes.centroid.x,
        'y': tract_shapes.centroid.y,
        'area': tract_shapes.area / 1000**2,  # sq km
        'geoid': tract_shapes.GEOID.apply(lambda x: str(x).zfill(11)),
    })

    demo_data = pd.read_csv(os.path.join(constants.TRACT_DATA_PATH, state_fips, 'DP02.csv'),
                            low_memory=False)
    demo_data['GEOID'] = demo_data.apply(
        lambda x: str(x['state']).zfill(2) + str(x['county']).zfill(3) + str(x['tract']).zfill(6),
        axis=1
    )
    demo_data = demo_data.set_index('GEOID')
    pop_data = demo_data['DP02_0122E'].to_dict()

    state_df['population'] = state_df.apply(lambda x: pop_data[x['geoid']], axis=1)

    shape_list = tract_shapes.geometry.to_list()
    adj_graph = pysal.lib.weights.Rook.from_iterable(shape_list).to_networkx()
    edge_dists = dict(nx.all_pairs_shortest_path_length(adj_graph))

    centroids = state_df[['x', 'y']].values
    plengths = squareform(pdist(centroids))

    save_path = os.path.join(constants.OPT_DATA_PATH, state_fips)
    os.makedirs(save_path, exist_ok=True)

    state_df.to_csv(os.path.join(save_path, 'state_df.csv'), index=False)
    np.save(os.path.join(save_path, 'lengths.npy'), plengths)
    nx.write_gpickle(adj_graph, os.path.join(save_path, 'G.p'))
    pickle.dump(edge_dists, open(os.path.join(save_path, 'edge_dists.p'), 'wb'))


def save_preprocessing(raw_data_paths,
                       save_path,
                       require_adjacency=False):
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
