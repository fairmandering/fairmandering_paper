import os
import pickle
import math
import numpy as np
from scipy.spatial.distance import pdist
from gerrypy import constants

experiments_to_dedup = [
    'experiment_results_NC_generation_1593060906',
    'experiment_results_IL_generation_1593125939'
]


def average_entropy(M):
    return (- M * np.ma.log(M).filled(0) - (1 - M) *
            np.ma.log(1 - M).filled(0)).sum() / (M.shape[0] * M.shape[1])


def svd_entropy(sigma):
    sigma_hat = sigma / sigma.sum()
    entropy = - (sigma_hat * (np.ma.log(sigma_hat).filled(0) / np.log(math.e))).sum()
    return entropy / (math.log(len(sigma)) / math.log(math.e))


deduped_results = {}
for exp in experiments_to_dedup:
    deduped_results[exp] = {}
    for trial in os.listdir(exp):
        trial_file = os.path.join(exp, trial)

        trial_data = np.load(trial_file, allow_pickle=True)[()]
        precinct_district_matrix = trial_data['precinct_district_matrix']
        n_districts = trial_data['trial_config']['n_districts']

        n_generated_districts = precinct_district_matrix.shape[1]
        precinct_district_matrix = np.unique(precinct_district_matrix, axis=1)
        n_unique_districts = precinct_district_matrix.shape[1]
        max_rank = min(precinct_district_matrix.shape)
        U, Sigma, Vt = np.linalg.svd(precinct_district_matrix)

        Dsim = 1 - pdist(precinct_district_matrix.T, metric='jaccard')

        precinct_coocc = precinct_district_matrix @ precinct_district_matrix.T
        precinct_conditional_p = precinct_coocc / precinct_district_matrix.sum(axis=1)

        L = (np.diag(precinct_coocc.sum(axis=1)) - precinct_coocc)
        D_inv = np.diag(precinct_coocc.sum(axis=1) ** -.5)
        e = np.linalg.eigvals(D_inv @ L @ D_inv)

        conditional_entropy = average_entropy(precinct_conditional_p)

        metrics = {
            'n_unique_districts': n_unique_districts,
            'p_duplicates': (n_generated_districts - n_unique_districts) / n_generated_districts,
            'conditional_entropy': conditional_entropy,
            'average_district_sim': n_districts * np.average(Dsim),
            'svd_entropy': svd_entropy(Sigma),
            '50p_approx_rank': sum(np.cumsum(Sigma / Sigma.sum()) < .5) / max_rank,
            '95p_approx_rank': sum(np.cumsum(Sigma / Sigma.sum()) < .95) / max_rank,
            '99p_approx_rank': sum(np.cumsum(Sigma / Sigma.sum()) < .99) / max_rank,
            'lambda_2': e[1],
            'lambda_k': e[n_districts],
        }
        deduped_results[exp][trial_file] = metrics

pickle.dump(deduped_results, open('deduped_results.p', 'wb'))
