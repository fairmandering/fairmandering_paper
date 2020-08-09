from scipy.spatial.distance import squareform
from scipy.stats import t
import numpy as np


def edge_dist_dict_to_matrix(edge_dists):
    dist_list = [edge_dists[i][j] for i in range(len(edge_dists))
                 for j in range(i+1, len(edge_dists))]
    dist_matrix = squareform(dist_list)
    np.fill_diagonal(dist_matrix, 1)
    return dist_matrix


def spatial_affiliation(state_df, hop_dist_matrix, alpha=2):
    inv_dist_matrix = 1 / hop_dist_matrix**alpha
    population_vector = state_df.population.values
    affiliation_vector = state_df[['2008', '2012', '2016']].values.mean(axis=1)
    std_vector = state_df[['2008', '2012', '2016']].values.std(axis=1, ddof=1)
    weight_norms = inv_dist_matrix @ population_vector
    affiliation = inv_dist_matrix @ np.multiply(population_vector, affiliation_vector)
    affiliation /= weight_norms
    spatial_std = inv_dist_matrix @ np.multiply(population_vector, std_vector)
    spatial_std /= weight_norms
    return affiliation, spatial_std, weight_norms


def spatial_deviation(state_df, hop_dist_matrix, alpha=2):
    mu_aff, mu_std, weight_norms = spatial_affiliation(state_df, hop_dist_matrix, alpha)
    population_vector = state_df.population.values
    affiliation_vector = state_df[['2008', '2012', '2016']].values.mean(axis=1)
    std_vector = state_df[['2008', '2012', '2016']].values.std(axis=1, ddof=1)

    tract_maj_p = 1 - t.cdf(.5, df=2, loc=mu_aff, scale=mu_std)

    maj_p = 1 - t.cdf(.5, df=2, loc=affiliation_vector, scale=std_vector)

    dif_mat = tract_maj_p[:, np.newaxis] - maj_p
    inv_dist_matrix = 1 / hop_dist_matrix
    spatial_weights = inv_dist_matrix ** alpha @ population_vector
    spatial_seat_sq_differnce = dif_mat ** 2 @ inv_dist_matrix ** alpha @ population_vector
    spatial_seat_std = (spatial_seat_sq_differnce / spatial_weights.sum()) ** .5

    return spatial_seat_std


def spatial_seat_entropy(affiliation, spatial_std):
    maj_p = 1 - t.cdf(.5, df=2, loc=affiliation, scale=spatial_std)
    binary_entropy = -maj_p * np.log(maj_p) - (1 - maj_p) * np.log(1 - maj_p)
    return binary_entropy


def entropy_over_alpha(state_df, hop_dist_matrix, average=False, step_size=.1, r=(0, 10)):
    alphas = np.arange(r[0], r[1] + step_size, step_size)
    entropy = []
    for a in alphas:
        affiliation, spatial_std, weight_norms = spatial_affiliation(state_df, hop_dist_matrix, alpha=a)
        binary_entropy = spatial_seat_entropy(affiliation, spatial_std)
        if average:
            binary_entropy = np.average(binary_entropy, weights=weight_norms)
        entropy.append(binary_entropy)
    return np.array(entropy), alphas


def stats_over_alpha(state_df, hop_dist_matrix, average=False, step_size=0.1, r=(0, 10)):
    alphas = np.arange(r[0], r[1] + step_size, step_size)
    win_ps = []
    entropy = []
    for a in alphas:
        affiliation, spatial_std, weight_norms = spatial_affiliation(state_df, hop_dist_matrix, a)
        maj_p = 1 - t.cdf(.5, df=2, loc=affiliation, scale=spatial_std)
        binary_entropy = -maj_p * np.log(maj_p) - (1 - maj_p) * np.log(1 - maj_p)
        if average:
            maj_p = np.average(maj_p, weights=weight_norms)
            binary_entropy = np.average(binary_entropy, weights=weight_norms)
        win_ps.append(maj_p)
        entropy.append(binary_entropy)
    return np.array(win_ps), np.array(entropy), alphas



