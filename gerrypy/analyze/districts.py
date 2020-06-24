import math
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial.distance import cdist


def roeck_compactness(districts, state_df, lengths):
    compactness_scores = []
    for d in districts:
        area = state_df.loc[d]['area'].sum()
        radius = lengths[np.ix_(d, d)].max() / 2000
        circle_area = radius ** 2 * math.pi
        roeck = area / circle_area
        compactness_scores.append(roeck)
    return compactness_scores


def dispersion_compactness(districts, state_df):
    compactness_scores = []
    for d in districts:
        population = state_df.loc[d]['population'].values
        dlocs = state_df.loc[d][['x', 'y']].values
        centroid = np.average(dlocs, weights=population, axis=0)
        geo_dispersion = (np.subtract(dlocs, centroid)**2).sum(axis=1)**.5 / 1000
        dispersion = np.average(geo_dispersion, weights=population)
        compactness_scores.append(dispersion)
    return compactness_scores


def diversity(districts, n_blocks):
    precinct_district_matrix = np.zeros((n_blocks, len(districts)))
    for ix, d in enumerate(districts):
        precinct_district_matrix[d, ix] = 1

    Dsim = precinct_district_matrix.T @ precinct_district_matrix
    Psim = precinct_district_matrix @ precinct_district_matrix.T



