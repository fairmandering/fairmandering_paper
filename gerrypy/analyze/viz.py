import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pysal.lib.weights import Queen
from matplotlib.colors import LinearSegmentedColormap as LSC
import matplotlib.pyplot as plt


def color_map(gdf, districting):
    # Takes a few seconds

    # block : distr num
    inv_map = {block: k for k, district in districting.items()
               for block in district}

    gdf['district'] = pd.Series(inv_map)
    shapes = []
    for name, group in gdf.groupby('district'):
        shapes.append(group.geometry.unary_union)
    shape_series = gpd.GeoSeries(shapes)
    G = Queen(shapes).to_networkx()
    color_series = pd.Series(nx.greedy_color(G))
    n_colors = len(set(color_series.values))

    cmap = LSC.from_list("", ["red", "lime", "dodgerblue",
                              'yellow', 'darkviolet', 'chocolate'][:n_colors])

    map_gdf = gpd.GeoDataFrame({'geometry': shape_series,
                                'color': color_series})
    map_gdf.plot(column='color', figsize=(15, 15), cmap=cmap)
    return map_gdf


def color_synthetic_map(config, districting):
    h, w = config['synmap_config']['height'], config['synmap_config']['width']
    tmap = np.zeros((h, w))
    for ix, district in enumerate(list(districting.values())):
        for tract in district:
            tmap[tract // w, tract % w] += ix
    plt.matshow(tmap)


def politics_map(gdf, politics, districting):
    pass
