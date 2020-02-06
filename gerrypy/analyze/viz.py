import geopandas as gpd
import networkx as nx
import pandas as pd
from pysal.lib.weights import Queen

def color_map(gdf, mapping):
    # Takes a few seconds

    # block : distr num
    inv_map = {block: k for k, district in mapping.items()
               for block in district}

    gdf['district'] = pd.Series(inv_map)
    shapes = []
    for name, group in  gdf.groupby('district'):
        shapes.append(group.geometry.unary_union)
    shape_series = gpd.GeoSeries(shapes)
    G = Queen(shapes).to_networkx()
    color_series = pd.Series(nx.greedy_color(G))
    if len(set(color_series.values)) == 3:
        cmap = 'jet'
    else:
        cmap = 'gist_rainbow'
    map_gdf = gpd.GeoDataFrame({'geometry': shape_series,
                                'color': color_series})
    map_gdf.plot(column='color', figsize=(15, 15), cmap=cmap)