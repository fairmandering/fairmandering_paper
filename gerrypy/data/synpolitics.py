import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import random


def demo_fn(synmap, mean=0, var=0.25):
    demo_distr = synmap + np.random.normal(loc=mean,
                                           scale=var,
                                           size=synmap.shape)
    return np.clip(demo_distr, 0, 1)


def create_fake_demographics(politics_config, synmap):
    synmap = synmap.flatten()
    demo_x = demo_fn(synmap)
    demo_y = demo_fn(synmap)
    demo_z = demo_fn(synmap)


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
    np.fill_diagonal(covar_mat, 0)
    variance = np.diag(np.random.normal(loc=0, scale=var, size=size))
    covar_mat += variance

    return mean_vector, covar_mat