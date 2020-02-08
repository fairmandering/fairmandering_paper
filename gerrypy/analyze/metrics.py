import numpy as np
import pandas as pd


def community_separation_index(districting, gdf):
    pass


def outcome_variance(districting, gdf):
    pass


def percent_duplicate_columns(columns):
    col_set = [frozenset(d) for d in columns]
    return (1 - len(set(col_set))) / len(col_set) * 100


def efficiency_gap(districting, state_df):
    dem_lost_votes = 0
    rep_lost_votes = 0
    dem_surplus_votes = 0
    rep_surplus_votes = 0
    all_total = 0

    for _, district in districting.items():
        d_votes = (state_df.loc[district].affiliation.values).dot(
            state_df.loc[district].population.values
        )
        r_votes = (1 - state_df.loc[district].affiliation.values).dot(
            state_df.loc[district].population.values
        )
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

    eff_gap_dict = {
        'dem_lost_votes': dem_lost_votes,
        'dem_surplus_votes': dem_surplus_votes,
        'rep_lost_votes': rep_lost_votes,
        'rep_surplus_votes': rep_surplus_votes,
        'gap': gap
    }

    return eff_gap_dict


def population_gap(districting, state_df):
    population = state_df.population.values
    average = population.sum() / len(districting)
    populations = []
    for d, district in districting.items():
        populations.append(population[district].sum())

    max_pop = max(populations)
    min_pop = min(populations)

    return min_pop / average, max_pop / average


def convex_compactness(districting, gdf):
    pass


def dispersion_compactness(districting, gdf):
    pass


def competitiveness(districting, state_df,
                    scale=((2, 'tossup'), (5, 'lean'),
                           (10, 'strong lean'), (50, 'likely'))):

    def helper(mean, scale):
        for percent, competition_rank in scale:
            if abs(mean - 50) < percent:
                party = 'democrat' if mean - 50 > 0 else 'republican'

                return competition_rank, party

    competitive_results = {v: {'democrat': 0, 'republican': 0}
                           for _, v in scale}
    for d, district in districting.items():
        pop_array = state_df.population.values[district]
        mean = state_df.affiliation.values[district].dot(pop_array) \
               / pop_array.sum() * 100
        competition_rank, party = helper(mean, scale)
        competitive_results[competition_rank][party] += 1

    return competitive_results




def n_competitive(districting, state_df, percent_threshold=5):
    competitive = 0
    for d, district in districting.items():
        pop_array = state_df.population.values[district]
        mean = state_df.affiliation.values[district].dot(pop_array) \
               / pop_array.sum() * 100
        if abs(50 - mean) < percent_threshold:
            competitive += 1
    return competitive

def show_metrics(districting, state_df):
    print(efficiency_gap(districting, state_df))
    print(competitiveness(districting, state_df))
    print(population_gap(districting, state_df))
