import numpy as np


def simulate_elections(district_map, state_df, political_covar, n_elections):
    n_seats = len(district_map)
    political_mean = state_df['affiliation'].values
    population = state_df['population'].values

    covar_cholesky = np.linalg.cholesky(political_covar)
    results = []
    for _ in range(n_elections):
        # election_sample = np.random.multivariate_normal(political_mean,
        #                                                 political_covar)
        election_sample = political_mean + covar_cholesky \
                          @ np.random.standard_normal(political_mean.size)

        n_dem_seats = 0
        n_dem_voters = 0
        for k, district in district_map.items():
            n_dem_votes = np.dot(election_sample[district],
                                 population[district])
            n_dem_voters += n_dem_votes
            vote_share = n_dem_votes / population[district].sum()

            if vote_share > .5:
                n_dem_seats += 1

        results.append(((n_dem_seats / n_seats),
                       n_dem_voters / population.sum()))

    return results
