import numpy as np
import scipy.stats as st

# TODO: expand
#

def expected_rep_gap(district, pop, mean, covar):
    weight = pop[district] / pop[district].sum()
    district_mean = mean[district].dot(weight)

    district_covar = covar[np.ix_(district, district)]
    district_variance = weight.T.dot(district_covar.dot(weight))

    z_value = (.5 - district_mean) / district_variance**.5

    win_p = 1 - st.norm.cdf(z_value)
    return win_p - district_mean



