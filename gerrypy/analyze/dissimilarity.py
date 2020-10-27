import pickle
from gerrypy.data.load import *
from gerrypy.analyze.viz import *
import numpy as np
import pandas as pd
import itertools

# Main Functions

def k_most_dissimilar(plans, num_blocks, k):
    """ Given a set of plans represented as a list of list of lists,
    returns the k most dissimilar plans
    Args:
        plans: a set of maps
        num_blocks: the number of blocks in the state
        k: the number of most distant maps to return
    """
    maps = convert_plans(plans, num_blocks)
    dist_matrix = compute_dist_matrix(maps)
    selected_maps = furthest_maps(dist_matrix, maps, k)
    return convert_back(selected_maps)


# Helper Functions
def convert_plans(plans, num_blocks):
    """
    Converts plans from a list of maps, each of which contain a list of districts, each of
    which contains a list of blocks in that district into a 3D binary numpy tensor,
    with the first dimension representing the number of maps, the second dimension representing
    the number of districts, and the third dimension a binary array of length (num_blocks),
    where a 1 in the ith entry indicates that the district contains block i.
    Args:
        plans: the list of maps to convert
        num_blocks: the largest number of blocks in a single district. Used for the conversion
        to binary arrays.
    """
    possible_districts = pd.Series(np.arange(0, num_blocks))
    converted_plans = []
    for i in range(len(plans)):
        plan = plans[i]
        converted_plan = []
        for j in range(len(plan)):
            district = plan[j]
            converted_plan.append(possible_districts.isin(district))
        converted_plans.append(converted_plan)
    return np.array(converted_plans).astype(int)


def compute_dist_matrix(maps):
    """
    Given a set of maps, return distance matrix
    Args:
      maps: (ndarray) a 3 dimensional tensor constructed using convert_plans
    """
    transposed_maps = np.transpose(maps, (0, 2,1))
    dimension = maps.shape[0]
    dist_matrix = np.zeros((dimension, dimension))
    count = 0
    i = 0
    j = 0


    for two_maps in itertools.product(maps, transposed_maps):
        if i >= j:
            dist = two_map_distance(two_maps[0], two_maps[1])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist
        count += 1
        if count % dimension == 0:
            i += 1
        j = count % dimension

    return dist_matrix

def two_map_distance(map_1, map_2):
    """
    Return the distance between map_1 and map_2
    Args:
        map_1: (ndarray) 2 Dimensional ndarray representing a map
        map_2: (ndarray) 2 Dimensional ndarray representing a map
    """
    new_map = map_1 @ map_2
    return 1000 * (1 / np.mean(np.max(new_map, 1)))


def furthest_maps(dist_matrix, maps, num_maps):
    """
    Given a distance matrix of size maps.shape[0] by maps.shape[0],
    a 3-d tensor representing the set of maps, and num_maps to return,
    returns num_maps most distant maps in the set of maps
    Args:
        dist_matrix: distance matrix
        maps: the set of maps
        num_maps: the number of maps to return which are most distant
    num_maps: (int) the number of maps to return
    """
    dimension = maps.shape[0] # Number of total maps
    output_indices = []
    output_maps = []

    flat_max = np.argmax(dist_matrix)
    abs_max = np.unravel_index(flat_max, dist_matrix.shape)

    output_indices.append(abs_max[0])
    output_indices.append(abs_max[1])
    output_maps.append(maps[abs_max[0]])
    output_maps.append(maps[abs_max[1]])

    for i in range(2, num_maps): # until we have the number of maps we need
        sum_distances = []
        for j in range(0, dimension): # For every map
            sum_distances.append(dist_to_optimal(
                dist_matrix, output_indices, j))
        sum_distances = np.asarray(sum_distances)
        next_max = np.argmax(sum_distances)
        output_indices.append(next_max)
        output_maps.append(maps[next_max])

    return np.asarray(output_maps)

def dist_to_optimal(dist_matrix, output_indices, elem):
    """
    Compute the distance from elem to every map in output_maps
    and then sum the result
    """
    dist = 0
    for i in range(0, len(output_indices)):
        dist += dist_matrix[output_indices[i], elem]
    return dist

def convert_back(plans):
    """
    Converts a set of plans in binary form back to regular form as a list of list of lists
    """
    num_plans = plans.shape[0]
    num_districts = plans.shape[1]
    converted_plans = []
    for i in range(num_plans):
        converted_plan = []
        plan = plans[i]
        for j in range(num_districts):
            district = plan[j]
            district = np.where(district)[0]
            district[district != 0]
            converted_plan.append(district.tolist())
        converted_plans.append(converted_plan)
    return converted_plans


def compute_difference(dist_matrix, selected_maps):
    """
    Helper function to evaluate how different average maps are from each
    other compared to the num_maps most distant maps
    """

    new_dist_matrix = compute_dist_matrix(selected_maps)
    dim = new_dist_matrix.shape[0]

    avg_new = 0
    for i in range(0,dim):
        for j in range(0, dim):
            if i != j:
                val = new_dist_matrix[i][j]
                avg_new += val

    avg_new = avg_new / (dim ** 2 - dim)

    dim = dist_matrix.shape[0]

    avg = 0
    for i in range(0, dim):
        for j in range(0, dim):
            if i != j:
                avg += dist_matrix[i][j]

    avg = avg / (dim ** 2 - dim)
    print(avg_new)
    print(avg)

    return avg_new / avg
