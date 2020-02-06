import numpy as np
import networkx as nx
import os
import pysal
import pandas as pd
import geopandas as gpd
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from gerrypy.data.data_matching import annotate_precincts


def geo_to_euclidean_coords(lat, lon):
    lon = np.deg2rad(lon)
    lat = np.deg2rad(lat)
    R = 6373
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return x, y, z


def geo_preprocess(gdf):
    x, y, z = geo_to_euclidean_coords(gdf['center_y'], gdf['center_x'])
    state_df = pd.DataFrame({'x': x, 'y': y, 'z': z})

    polygons = gdf.geometry.values
    adjacency_graph = pysal.lib.weights.Rook(polygons).to_networkx()

    return state_df, adjacency_graph


# def collect_paths(state):



def save_preprocessing(raw_data_paths, save_path):
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
    print('Making adjacency graph')
    precincts = gpd.read_file(precinct_shape_path).to_crs(epsg=4326)
    precincts['center_x'] = precincts.centroid.x
    precincts['center_y'] = precincts.centroid.y

    state_df, adjacency_graph = geo_preprocess(precincts)

    state_df['population'] = population
    state_df['affiliation'] = mean

    # Save data necessary to run redistricting
    print('Saving...')
    np.save(os.path.join(save_path, 'covar'), sigma)
    state_df.to_csv(os.path.join(save_path, 'state_df.csv'), index=False)
    nx.write_gpickle(adjacency_graph, os.path.join(save_path, 'G.p'))

