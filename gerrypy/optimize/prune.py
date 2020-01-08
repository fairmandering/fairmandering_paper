import numpy as np
import random
from gerrypy.utils.spatial_utils import vecdist


def yi_prune(lengths, ratio):
    return {k: v for k, v in lengths.items() if random.random() > ratio}


def make_lengths_data(config, state_df):
    tracts = list(state_df.index)
    x = state_df['x'].values
    y = state_df['y'].values
    pop = state_df['population'].values
    prune_radius = config['init_config']['population_pruning_radius']
    n_districts = config['n_districts']
    ideal_pop = prune_radius * np.sum(pop) / n_districts

    tract_lengths_dict = {}
    for ix, tract in enumerate(tracts):
        lens = xij_preprocess(tracts, ix, pop, x, y, ideal_pop,
                              config['euclidean'])
        tract_lengths_dict[tract] = lens

    return tract_lengths_dict


def xij_preprocess(tracts, t, pop, cent_x, cent_y, ideal_pop, is_euclidean):
    if is_euclidean:
        pdist = np.linalg.norm(np.stack([cent_x - cent_x[t],
                                         cent_y - cent_y[t]]),
                               axis=0)
    else:
        pdist = vecdist(cent_y, cent_x, cent_y[t], cent_x[t]).flatten()
    if ideal_pop/2 > np.sum(pop) - 100:
        return {tracts[tr]: int(pdist[tr]) for tr in range(len(tracts))}
    search_range = [0, max(pdist)]
    search_radius = sum(search_range) / 2
    in_range = np.sum(pop[pdist < search_radius])
    iters = 0
    while in_range < ideal_pop or in_range > ideal_pop * 1.05:
        if in_range < ideal_pop:
            search_range[0] = sum(search_range) / 2
        else:
            search_range[1] = sum(search_range) / 2
        if iters > 30:
            break
        search_radius = sum(search_range) / 2
        in_range = np.sum(pop[pdist < search_radius])
        iters += 1
    tract_ix_in_range = np.argwhere(pdist < search_radius)
    return {tracts[tr]: pdist[tr] for tr in tract_ix_in_range.flatten()}