import pandas as pd
import numpy as np
from scipy.stats import t
from gerrypy.analyze.districts import dispersion_compactness


def create_district_df(bdm, state_df):
    district_list = []
    sum_columns = ['area', 'population']
    bdm = bdm.astype(bool)
    for row in bdm.T:
        district_df = state_df.iloc[row]
        distr_series = pd.Series(np.average(district_df, weights=district_df['population'], axis=0),
                                 index=state_df.columns)
        distr_series[sum_columns] = district_df[sum_columns].sum()
        district_list.append(distr_series)
    return pd.DataFrame(district_list)


def enumerate_distribution(plans, bdm, state_df, type):
    if type == 'compactness':
        district_blocks = [np.nonzero(d)[0] for d in bdm.T]
        d_compactness = np.array(dispersion_compactness(district_blocks, state_df))
        return d_compactness[np.array(plans)].mean(axis=1)
    elif type == 'politics':
        district_df = create_district_df(bdm, state_df)
        mu = district_df[['2008', '2012', '2016']].mean(axis=1).values
        std = np.maximum(district_df[['2008', '2012', '2016']].std(axis=1), .04).values
        p_win = 1 - t.cdf(.5, df=2, loc=mu, scale=std)
        expected_diff = p_win - mu
        return expected_diff[np.array(plans)].sum(axis=1)
    else:
        raise ValueError('Invalid distribution type')
