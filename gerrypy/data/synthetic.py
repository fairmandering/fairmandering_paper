import random
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import rbf_kernel
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import pdist, squareform


def generate_synthetic_input(config):
    synmap_config = config['synmap_config']
    syn_map = generate_map(synmap_config)
    h, w = synmap_config['height'], synmap_config['width']
    G = nx.grid_graph([w, h])
    G = nx.convert_node_labels_to_integers(G)
    pop_array = syn_map.flatten() * 100
    x = np.arange(w).repeat(h).reshape(w, h).T.flatten()
    y = np.arange(h).repeat(w).reshape(h, w).flatten()
    state_df = pd.DataFrame({'population': pop_array,
                             'x': x, 'y': y})
    state_poli_mean, state_covar = create_political_distribution(config,
                                                                 state_df)
    state_df['affiliation'] = state_poli_mean
    lengths = squareform(pdist(state_df[['x', 'y']].values))

    return state_df, G, state_covar, lengths


def create_political_distribution(config, state_df):
    mean = config['politics_config']['noise_mean']
    var = config['politics_config']['noise_variance']
    population = state_df.population.values
    size = len(population)
    normal_synmap = population / np.max(population)
    mean_vector = normal_synmap + np.random.normal(loc=mean,
                                                   scale=var,
                                                   size=size)
    mean_vector = np.clip(mean_vector, 0, 1)

    X = state_df[['population', 'x', 'y']].values

    gamma = config['politics_config']['covariance_gamma']
    covar_mat = rbf_kernel(X, gamma=gamma)
    np.fill_diagonal(covar_mat, 1)
    variance = np.abs(np.diag(np.random.normal(loc=0, scale=var, size=size)))
    covar_mat += variance

    return mean_vector, covar_mat


def gkern(size, sig):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((size, size))
    # set element at the middle to one, a dirac delta
    inp[size // 2, size // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return gaussian_filter(inp, sig)


def generate_map(config):
    h = config['height']
    w = config['width']
    vote_map = np.zeros((h, w)) + .5
    urban_area = np.zeros((h, w))
    for i in range(config['n_cities']):
        y, x, size, vote_dist = generate_city(config)
        vote_map[y:y + size, x:x + size] += vote_dist
        urban_area[y:y + size, x:x + size] = 1

    vote_map = np.minimum(vote_map, 1)

    rural_area = 1 - urban_area
    rural_pop = np.sum(rural_area)
    urban_pop = np.sum(urban_area)

    urban_votes = np.mean(vote_map[np.where(urban_area)]) * urban_pop
    rural_votes = ((config['dem_vote'] * h * w) - urban_votes)
    rural_mean = (rural_votes / rural_pop)

    vote_map[np.where(rural_area)] = np.random.normal(rural_mean,
                                                      config['rural_vote_std'],
                                                      len(np.where(rural_area)[0]))

    vote_map = np.maximum(vote_map, 0)

    assert np.all(vote_map <= 1) and np.all(vote_map >= 0)

    return vote_map


def generate_city(config):
    h = config['height']
    w = config['width']
    y, x = random.randint(0, h - 2), random.randint(0, w - 2)
    size = int(np.random.exponential(config['exponential_scale'])) + 1
    if size % 2 == 0:
        size += 1
    size = min(size, min(h - y, w - x))
    if size % 2 == 0:
        size -= 1
    if size > 1:
        scale = size ** 2 * random.uniform(config['scale_bounds'][0],
                                           config['scale_bounds'][1])
        sigma = size * random.uniform(config['sigma_bounds'][0],
                                      config['sigma_bounds'][1])
        vote_dist = scale * gkern(size, sigma)
    else:
        vote_dist = np.array([[random.uniform(0, .4)]])
    return y, x, size, vote_dist