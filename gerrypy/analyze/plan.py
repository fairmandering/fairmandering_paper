def responsiveness():
    raise NotImplementedError


def symmetry():
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
