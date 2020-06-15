import math
import random
import numpy as np


class SHPNode:
    def __init__(self, n_districts, area, is_root=False):
        self.is_root = is_root
        self.n_districts = n_districts
        self.area = area
        self.children_ids = []

        self.pepper = random.randint(-1e10, 1e10)
        self.area_hash = hash(frozenset(area))
        self.id = self.area_hash + self.pepper

        self.n_sample_failures = 0
        self.n_infeasible_samples = 0

    def sample_n_splits_and_child_sizes(self, config):
        n_distrs = self.n_districts
        n_splits = random.randint(min(config['min_n_splits'], n_distrs),
                                  min(config['max_n_splits'], n_distrs))

        ub = max(math.ceil(config['max_split_population_difference']
                           * n_distrs / n_splits), 2)
        lb = max(math.floor((1 / config['max_split_population_difference'])
                            * n_distrs / n_splits), 1)

        child_n_distrs = np.zeros(n_splits) + lb
        while int(sum(child_n_distrs)) != n_distrs:
            ix = random.randint(0, n_splits - 1)
            if child_n_distrs[ix] < ub:
                child_n_distrs[ix] += 1

        return child_n_distrs

    def __repr__(self):
        print_str = "Node %d \n" % self.id
        internals = self.__dict__
        for k, v in internals.items():
            if k == 'area':
                continue
            print_str += k + ': ' + v.__repr__() + '\n'
        return print_str


class SampleTree:
    def __init__(self, config, n_districts, level=0):
        self.n_districts = n_districts
        self.level = 0

        if n_districts > 1:
            children_n_distrs = SHPNode.sample_n_splits_and_child_sizes()
            self.children = [SampleTree(config, n, level + 1)
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
