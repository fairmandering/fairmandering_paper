import numpy as np
import networkx as nx
import os
import pysal
import pandas as pd
import geopandas as gpd
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from gerrypy.data.data_matching import annotate_precincts
from gerrypy.data.adjacency import create_adjacency_graph


def save_preprocessing(raw_data_paths,
                       save_path,
                       require_adjacency=False):
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

    kernel = RBF() + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel)

    gp.fit(X, y)
    mean, sigma = gp.predict(X, return_cov=True)

    # Create state_df and adjacency graph
    precincts = gpd.read_file(precinct_shape_path).to_crs("EPSG:3078")  # m units

    state_df = pd.DataFrame({'x': precincts.centroid.x,
                             'y': precincts.centroid.y})

    if require_adjacency:
        print('Making adjacency graph')
        adjacency_graph = create_adjacency_graph(precincts)

    state_df['population'] = population
    state_df['affiliation'] = np.clip(mean, 0, 1)

    # Save data necessary to run redistricting
    print('Saving...')
    np.save(os.path.join(save_path, 'covar'), sigma)
    state_df.to_csv(os.path.join(save_path, 'state_df.csv'), index=False)
    if require_adjacency:
        nx.write_gpickle(adjacency_graph, os.path.join(save_path, 'G.p'))

