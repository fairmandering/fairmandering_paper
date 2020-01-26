import math
import random
import numpy as np


def sample_n_children(hconfig, n_distrs):
    n_splits = random.randint(min(hconfig['min_n_splits'], n_distrs),
                              min(hconfig['max_n_splits'], n_distrs))

    ub = max(math.ceil(hconfig['max_split_population_difference']
                       * n_distrs / n_splits), 2)
    lb = max(math.floor((1 / hconfig['max_split_population_difference'])
                        * n_distrs / n_splits), 1)

    child_n_distrs = np.zeros(n_splits) + lb
    while int(sum(child_n_distrs)) != n_distrs:
        ix = random.randint(0, n_splits - 1)
        if child_n_distrs[ix] < ub:
            child_n_distrs[ix] += 1

    return child_n_distrs


class SampleTree:
    def __init__(self, hconfig, n_districts, level=0):
        self.n_districts = n_districts
        self.level = 0

        if n_districts > 1:
            children_n_distrs = sample_n_children(hconfig, n_districts)
            self.children = [SampleTree(hconfig, n, level + 1)
                             for n in children_n_distrs]
        else:
            self.children = None

        self.max_levels_to_leaf = 0
        self.max_layer()

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.n_districts) + "\n"
        if self.children is not None:
            for child in self.children:
                ret += child.__repr__(level + 1)
        return ret

    def max_layer(self):
        try:
            to_leaf = 1 + max([child.max_layer() for child in self.children])
            self.max_levels_to_leaf = to_leaf
            return to_leaf
        except TypeError:
            return 0
