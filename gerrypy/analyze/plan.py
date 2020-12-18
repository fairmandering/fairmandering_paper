"""TODO: update with non county data election data."""

import numpy as np
from scipy.stats import t
from scipy.integrate import simps
from gerrypy.analyze.poibin import PoiBin


def seat_vote_curve_point_estimate(plan_df, perturb=0):
    mean = plan_df['mean'].values
    if perturb:
        mean += np.random.random(size=mean.size) * perturb
    xs = []
    ys = []
    for sigma in np.arange(-1, 1, .001):
        if sigma > 0:
            mu_sig = np.minimum(mean + sigma, 1)
        else:
            mu_sig = np.maximum(mean + sigma, 0)
        vote_share = np.sum(mu_sig) / len(plan_df)
        xs.append(vote_share)
        ys.append(np.sum(mu_sig > .5) / len(plan_df))
    return np.array(xs), np.array(ys)


def seat_vote_curve_t_estimate_with_seat_std(plan_df, step_size=.001):
    mean = plan_df['mean'].values
    std = plan_df['std_dev'].values
    DoF = plan_df['DoF'].values
    means_by_state_share = district_votes_given_state_vote(plan_df, step_size)
    std = np.tile(std, (means_by_state_share.shape[1], 1)).T
    win_p = 1 - t.cdf(.5, DoF, means_by_state_share, std)
    ys = win_p.mean(axis=0)
    xs = means_by_state_share.mean(axis=0)
    poission_binomial_variance = np.multiply(win_p, 1 - win_p).sum(axis=0)
    stds = poission_binomial_variance ** .5 / len(plan_df)
    return xs, ys, stds


def sample_elections(plan_df, n=1000, p_seats=False):
    elec_results = plan_df[['2008', '2012', '2016']].values
    state_year_results = elec_results.mean(axis=0)
    state_vote_share_mean = state_year_results.mean()
    state_vote_t = t(df=2, loc=state_vote_share_mean,
                     scale=state_year_results.std(ddof=1))

    state_vote_share_samples = state_vote_t.rvs(n)

    district_mean = elec_results.mean(axis=1)
    district_std = elec_results.std(axis=1, ddof=1)
    district_vote_t = t(df=2, loc=district_mean, scale=district_std)

    district_static_samples = district_vote_t.rvs((n, len(district_mean)))
    district_mean_samples = district_static_samples.mean(axis=1)
    district_vote_shares = (district_static_samples +
                            (state_vote_share_samples - district_mean_samples)[:, np.newaxis])

    if p_seats:
        seat_shares = np.sum(1 - t.cdf(.5, df=2, loc=district_vote_shares, scale=district_std), axis=1)
    else:
        seat_shares = np.sum(district_vote_shares > 0.5, axis=1)
    vote_shares = np.sum(district_vote_shares, axis=1)
    return seat_shares / len(plan_df), vote_shares / len(plan_df)


def district_votes_given_state_vote(plan_df, step_size=.001, tol=1e-5):
    past_elections = plan_df[['2008', '2012', '2016']].values
    means = past_elections.mean(axis=1)
    state_vote_share = np.arange(0, 1 + step_size, step_size)
    mu_sig = np.tile(means, (len(state_vote_share), 1)).T
    sig_min = -np.ones(len(state_vote_share))
    sig_max = np.ones(len(state_vote_share))
    while np.any(abs(mu_sig.mean(axis=0) - state_vote_share) > tol):
        lt = mu_sig.mean(axis=0) - state_vote_share < 0
        sig_min[lt] = ((sig_max + sig_min) / 2)[lt]
        sig_max[~lt] = ((sig_max + sig_min) / 2)[~lt]
        sigma = (sig_max + sig_min) / 2
        mu_sig = np.clip(means.reshape(means.shape + (1,)) + sigma, 0, 1)
    return mu_sig


def district_means_for_state_share(means, state_share, tol=1e-5):
    assert 0 <= state_share <= 1
    sig_min, sig_max = -1, 1
    mu_sig = means.copy()
    while abs(mu_sig.mean() - state_share) > tol:
        if mu_sig.mean() - state_share < 0:
            sig_min = (sig_max + sig_min) / 2
        else:
            sig_max = (sig_max + sig_min) / 2
        sigma = (sig_max + sig_min) / 2
        mu_sig = np.clip(means + sigma, 0, 1)
    return mu_sig


def estimate_responsiveness(plan_df):
    elec_results = plan_df[['2008', '2012', '2016']].values
    state_year_results = elec_results.mean(axis=0)
    state_vote_t = t(df=2, loc=state_year_results.mean(),
                     scale=state_year_results.std(ddof=1))
    truncation_factor = 1 / (state_vote_t.cdf(1) - state_vote_t.cdf(0))

    district_share = district_votes_given_state_vote(plan_df)
    state_share = district_share.mean(axis=0)

    district_std = elec_results.std(axis=1, ddof=1)
    district_std = np.tile(district_std, (len(state_share), 1)).T

    vote_seat_slope = np.nan_to_num(t.pdf(.5, df=2, loc=district_share, scale=district_std)).mean(axis=0)
    return simps(vote_seat_slope * state_vote_t.pdf(state_share) * truncation_factor, state_share)


def estimate_symmetry(plan_df):
    elec_results = plan_df[['2008', '2012', '2016']].values
    state_year_results = elec_results.mean(axis=0)
    state_vote_t = t(df=2, loc=state_year_results.mean(),
                     scale=state_year_results.std(ddof=1))
    pA_votes, pA_seats, stds = seat_vote_curve_t_estimate_with_seat_std(plan_df)
    pB_votes = (1 - pA_votes)[::-1]
    pB_seats = (1 - pA_seats)[::-1]

    truncation_factor = 1 / (state_vote_t.cdf(1) - state_vote_t.cdf(0))

    party_A_auc = simps(state_vote_t.pdf(pA_votes) * pA_seats * truncation_factor, pA_votes)
    party_B_auc = simps(state_vote_t.pdf(pB_votes) * pB_seats * truncation_factor, pB_votes)

    return party_A_auc - party_B_auc


def fifty_gap(plan_df):
    past_elections = plan_df[['2008', '2012', '2016']].values
    means = past_elections.mean(axis=1)
    std = past_elections.std(axis=1)
    adjusted_means = district_means_for_state_share(means, .5)
    win_p = 1 - t.cdf(.5, 2, adjusted_means, std)
    return win_p.sum() / len(plan_df) - .5


def majority_prob(plan_df):
    elec_results = plan_df[['2008', '2012', '2016']].values
    state_year_results = elec_results.mean(axis=0)
    state_vote_t = t(df=2, loc=state_year_results.mean(),
                     scale=state_year_results.std(ddof=1))
    truncation_factor = 1 / (state_vote_t.cdf(1) - state_vote_t.cdf(0))

    district_share = district_votes_given_state_vote(plan_df)
    state_share = district_share.mean(axis=0)

    district_std = elec_results.std(axis=1, ddof=1)
    district_std = np.tile(district_std, (len(state_share), 1)).T

    district_vote_maj = t.cdf(.5, df=2, loc=district_share, scale=district_std)
    success_prob = (1 - np.nan_to_num(district_vote_maj)).T
    seats_for_majority = len(plan_df) // 2
    majority_probability = [1-PoiBin(probabilities=maj_prob).cdf(seats_for_majority)
                            for maj_prob in success_prob]
    return simps(majority_probability * state_vote_t.pdf(state_share) * truncation_factor,
                 state_share)


def majority_majority_prob():
    raise NotImplementedError


def community_separation_index():
    raise NotImplementedError


def historical_seat_share(plan_df):
    return (plan_df[['2008', '2012', '2016']] < .5).sum(axis=0).mean()


def competitiveness(plan_df, threshold=.05):
    return (abs(plan_df[['2008', '2012', '2016']] - .5) < threshold).sum(axis=0).mean()


def efficiency_gap(plan_df):
    def year_estimate(year):
        dem_lost_votes = 0
        rep_lost_votes = 0
        dem_surplus_votes = 0
        rep_surplus_votes = 0
        all_total = 0

        for d, row in plan_df.iterrows():
            d_votes = (1 - row[year]) * row['population']
            r_votes = row[year] * row['population']
            total = d_votes + r_votes
            all_total += total
            winning_total = int(round(total / 2. + 1))
            dem_winner = d_votes > r_votes
            if dem_winner:
                rep_lost_votes += r_votes
                dem_surplus_votes += (d_votes - winning_total)
            else:
                dem_lost_votes += d_votes
                rep_surplus_votes += (r_votes - winning_total)

        dem_wasted = dem_lost_votes + dem_surplus_votes
        rep_wasted = rep_lost_votes + rep_surplus_votes
        gap = (dem_wasted - rep_wasted) / float(all_total)
        return gap

    gaps = []
    for y in ['2008', '2012', '2016']:
        gaps.append(year_estimate(y))
    return sum(gaps) / len(gaps)
