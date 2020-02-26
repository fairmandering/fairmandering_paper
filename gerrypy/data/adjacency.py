import numpy as np
import networkx as nx
import time
from scipy.spatial.distance import cdist, pdist, squareform


def create_adjacency_graph(gdf,
                           interp_scaling=5,
                           interp_offset=50,
                           centroid_search_fraction=.01,
                           adjacent_distance_threshold=50,
                           min_adjacent_boundary_pts=2,
                           large_component_searches=50,
                           component_edge_threshold_factor=1.5):
    """
    Constructs an adjacency graph from a geodataframe

    Note: Make sure the geodateframe is projected with units of feet or meters.
    For the USA, EPSG:3078 is recommended.

    The algorithm proceeds in 2 phases. It first constructs a set of higher
    granularity polygons by interpolating points on the edge (so straight
    edges don't only have 2 points) guided by the interpolation parameters
    [interp_scaling] and [interp_offset]. With the set of higher granularity
    polygons look at their centroids and iterate over each one,
    identifying the [centroid_search_fraction] closest other centroids.
    For each pair identify the number of boundary points within
    [max_adjacent_distance] of each other. If this number is greater than
    [min_adjacent_boundary_pts] then classify the pair as adjacent.

    After this first phase all nodes that are directly adjacent will be
    connected but the graph might contain more than one component (e.g. islands).
    To connect the components we iteratively attach the two nearest.
    Specifically, start by finding the pair of centroids from different
    components with minimum distance (d1). Then find the pair of boundary
    points in the two components with minimum distance (d2).
    Then for each polygon with centroid within an [island_search_factor] of d1
    add an edge if there is a pair of boundary points less than
    [island_edge_factor] * d2 between 2 polygons. Continue this process
    until there is only one component.

    Args:
        gdf (geopandas.GeoDataFrame): Contains the geometries of the polygons

        interp_scaling (float): Control the interpolation density of line
            segments in a polygon. Increase scale causes decreased density.
            See interp_pts()

        interp_offset (float): Control the minimum length of line segment
            where interpolation is used. See interp_pts()

        centroid_search_fraction (float): The fraction of total polygons to check
            connectivity for. Increasing this number will cause slower runtimes
            but setting it too low could lose accuracy if there is an adjacent
            polygon with a centroid that happens to be far away. Smaller states
            should be run with larger search fraction values.

        adjacent_distance_threshold (float): The maximum distance (measured in
            the units of the gdf (ft or m)) for two points to be considered
            connected. Increasing this number will increase the number of corner
            connected polygons but decreasing it increases the probability of
            a straight edge miss if the interpolated points are not dense enough.

        min_adjacent_boundary_pts (int): The minimum number of points that need
            to be connected before we consider two polygons to be adjacent.
            Opposite to [adjacent_distance_threshold] increasing this parameter
            will decrease the number of corner points but will make a straight
            edge miss more likely if there isn't a high enough interpolation
            density.

        large_component_searches (int): The number of polygons in the larger
            component to check for a inter-component edge. These are the
            closest polygons by centroids. Increasing this parameter
            requires more time but there is typically only a few component
            matching loops so it makes a small difference.

        component_edge_threshold_factor (float): After finding the minimum
            distance between two boundary points in two different components d2,
            add an edge between every 2 polygons that have minimum boundary
            point distance less than d2 * [component_edge_threshold_factor].
            Increasing this parameter will increase the connectivity of
            islands to the main land mass.


    Returns: undirected NetworkX graph

    Note: Edges added is phase 2 have edge attribute "inferred"
    """
    # Create finer granularity polygons
    interpolated_polygons = create_interpolated_polygons(gdf,
                                                         interp_scaling,
                                                         interp_offset)

    # Determine the polygons we will test for connectivity
    centroids = gdf.centroid
    centroids = np.vstack([centroids.values.x, centroids.values.y]).T
    centroid_dists = squareform(pdist(centroids))
    searches = np.argsort(centroid_dists)[:, 1:int(len(centroids)
                                                   * centroid_search_fraction)]

    G = nx.Graph()
    G.add_nodes_from(list(gdf.index))

    # Algorithm first phase: find adjacent polygons
    time_stamp = time.time()
    searched_pairs = set()
    for precinct_ix1, row in enumerate(searches):
        if precinct_ix1 % 100 == 0:
            print(precinct_ix1, 'nodes completed in',
                  time.time() - time_stamp, 's')
            time_stamp = time.time()
        precinct1 = interpolated_polygons[precinct_ix1]
        for precinct_ix2 in [p2 for p2 in row  # Take advantage of symmetry
                             if (precinct_ix1, p2) not in searched_pairs]:
            precinct2 = interpolated_polygons[precinct_ix2]
            point_dists = cdist(precinct1, precinct2)
            if np.sum(point_dists < adjacent_distance_threshold) \
                    >= min_adjacent_boundary_pts:
                G.add_edge(precinct_ix1, precinct_ix2)
            # Don't need to check the reverse pair
            searched_pairs.add((precinct_ix2, precinct_ix1))

    comps = [list(c) for c in list(nx.connected_components(G))]

    # Algorithm second phase: connect components
    while len(comps) > 1:
        min_component_dists = []
        # Find the pair of components with smallest minimum centroid distance
        for ix in range(len(comps) - 1):
            comp1 = comps[ix]
            for jx in range(ix + 1, len(comps)):
                comp2 = comps[jx]
                component_centroid_dists = centroid_dists[comp1, :][:, comp2]
                dist = np.min(component_centroid_dists)
                min_component_dists.append((ix, jx, dist))

        component_match = min(min_component_dists, key=lambda x: x[-1])
        ix, jx, min_dist = component_match

        print('Matching components with hash values', hash(frozenset(comps[0])),
              hash(frozenset(comps[1])))

        small_component = jx if len(comps[ix]) >= len(comps[jx]) else ix
        large_component = jx if len(comps[ix]) < len(comps[jx]) else ix

        # Find the closest polygons from the large component
        component_dists = centroid_dists[comps[small_component], :][:, comps[large_component]]
        large_comp_search_order = np.argsort(component_dists)[:, :large_component_searches]

        poly_min_dists = []
        for i, poly_ix in enumerate(comps[small_component]):
            poly1_pts = unwind_coords(gdf.geometry[poly_ix])
            for order_ix in large_comp_search_order[i]:
                poly_jx = comps[large_component][order_ix]
                poly2_pts = unwind_coords(gdf.geometry[poly_jx])
                poly_min_dist = np.min(cdist(poly1_pts, poly2_pts))
                poly_min_dists.append((poly_ix, poly_jx, poly_min_dist))

        closest_polys = min(poly_min_dists, key=lambda x: x[-1])
        edge_threshold = (closest_polys[-1] + 1) * component_edge_threshold_factor

        for poly_ix, poly_jx, dist in poly_min_dists:
            if dist < edge_threshold:
                G.add_edge(poly_ix, poly_jx, inferred=True)
                print('added edge', poly_ix, poly_jx)
        comps = [list(c) for c in list(nx.connected_components(G))]

    return G


def interp_pts(dist, scaling, offset):
    return np.power(np.maximum((dist - offset) / scaling, 0), 1 / 2)


def unwind_coords(poly):
    try:
        return np.array(poly.exterior.coords)
    except AttributeError:
        return np.concatenate([p.exterior.coords for p in poly])


def create_interpolated_polygons(precincts, interp_scaling, interp_offset):
    interpolated_polygons = []
    for ix, precinct in precincts.geometry.iteritems():
        poly_pts = []
        try:
            perimeter = np.array(precinct.exterior.coords)
            for pt1, pt2 in zip(perimeter[:-1], perimeter[1:]):
                line_len = np.linalg.norm(pt2 - pt1)
                n_interp_pts = int(interp_pts(line_len,
                                              interp_scaling,
                                              interp_offset)) + 1
                interp_dist = (pt2 - pt1) / n_interp_pts
                line_pts = [pt1 + interp_dist * i for i in range(n_interp_pts)]
                poly_pts += line_pts
        except AttributeError:  # Multipolygon case
            for polygon in list(precinct):
                perimeter = np.array(polygon.exterior.coords)
                for pt1, pt2 in zip(perimeter[:-1], perimeter[1:]):
                    line_len = np.linalg.norm(pt2 - pt1)
                    n_interp_pts = int(round(interp_pts(line_len, 5, 100))) + 1
                    interp_dist = (pt2 - pt1) / n_interp_pts
                    line_pts = [pt1 + interp_dist * i
                                for i in range(n_interp_pts)]
                    poly_pts += line_pts

        interpolated_polygons.append(np.array(poly_pts))
    return interpolated_polygons


if __name__ == '__main__':
    import os
    import geopandas as gpd

    ma = gpd.read_file(os.path.join('precincts', 'ma_2016_precincts'))
    create_adjacency_graph(ma[:200])
