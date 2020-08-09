import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pysal.lib.weights import Queen
from matplotlib.colors import LinearSegmentedColormap as LSC
import matplotlib.pyplot as plt
import seaborn as sns
from gerrypy.analyze.plan import *


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
    ax = map_gdf.plot(column='color', figsize=(15, 15), cmap=cmap, edgecolor='black', lw=.5)
    gdf.plot(ax=ax, facecolor='none', edgecolor='white', lw=.05)
    ax.axis('off')
    return map_gdf


def draw_adjacency_graph(gdf, G, size=(200, 150)):
    base = gdf.plot(color='white', edgecolor='black', figsize=size, lw=.5)
    edge_colors = ['green' if G[u][v].get('inferred', False) else 'red'
                   for u, v in G.edges]
    pos = {i: (geo.centroid.x, geo.centroid.y)
           for i, geo in gdf.geometry.iteritems()}
    if len(G) == len(gdf) + 1:  # If adj graph with dummy node
        pos[len(gdf)] = (min(gdf.centroid.x), min(gdf.centroid.y))
    nx.draw_networkx(G,
                     pos=pos,
                     ax=base,
                     node_size=1,
                     width=.5,
                     linewidths=.5,
                     with_labels=False,
                     edge_color=edge_colors)


def color_synthetic_map(config, districting):
    h, w = config['synmap_config']['height'], config['synmap_config']['width']
    tmap = np.zeros((h, w))
    for ix, district in enumerate(list(districting.values())):
        for tract in district:
            tmap[tract // w, tract % w] += ix
    plt.matshow(tmap)


# helper function to visualize data
def plot_percentiles(xs, ys):

    probs = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    percentiles = [np.percentile(ys, prob, axis=0) for prob in probs]

    ultra_light = '#ede1e1'
    light = "#DCBCBC"
    light_highlight = "#C79999"
    mid = "#B97C7C"
    mid_highlight = "#A25050"
    dark = "#8F2727"
    dark_highlight = "#7C0000"
    green = "#00FF00"

    plt.fill_between(xs, percentiles[0], percentiles[10],
                     facecolor=light, color=ultra_light)
    plt.fill_between(xs, percentiles[1], percentiles[9],
                     facecolor=light, color=light)
    plt.fill_between(xs, percentiles[2], percentiles[8],
                     facecolor=light_highlight, color=light_highlight)
    plt.fill_between(xs, percentiles[3], percentiles[7],
                     facecolor=mid, color=mid)
    plt.fill_between(xs, percentiles[4], percentiles[6],
                     facecolor=mid_highlight, color=mid_highlight)
    plt.plot(xs, percentiles[5], color=dark)


def politics_map(gdf, politics, districting):
    def result_to_color(vote):
        if vote < .4:
            return 0
        elif vote < .45:
            return 1
        elif vote < .49:
            return 2
        elif vote < .51:
            return 3
        elif vote < .55:
            return 4
        elif vote < .6:
            return 5
        else:
            return 6
    # Takes a few seconds

    # block : distr num
    inv_map = {block: k for k, district in districting.items()
               for block in district}

    gdf['district'] = pd.Series(inv_map)

    shapes = []
    colors = []
    for name, group in gdf.groupby('district'):
        shapes.append(group.geometry.unary_union)
        colors.append(result_to_color(politics[name]))
    shape_series = gpd.GeoSeries(shapes)

    # TODO: better colors
    color_map = ["#0000ffff", "#0000ff90", "#0000ff50", '#a00ff0f0', '#ff000050',
                              '#ff000090', '#ff0000ff']

    map_gdf = gpd.GeoDataFrame({'geometry': shape_series,
                                'color': pd.Series(colors)})
    ax = map_gdf.plot(color='none', figsize=(15, 15), edgecolor='black', lw=1)
    for name, group in map_gdf.groupby('color'):
        group.plot(ax=ax, color=color_map[name], edgecolor='black', lw=1)
    gdf.plot(ax=ax, facecolor='none', edgecolor='white', lw=.05)
    ax.axis('off')
    return map_gdf


def plot_seat_vote_curve(plan_df, n_samples=1000, height=10):
    xs, ys, stds = seat_vote_curve_t_estimate_with_seat_std(plan_df)
    seats, votes = sample_elections(plan_df, n=n_samples, p_seats=True)
    g = sns.jointplot(votes, seats, kind='kde', space=0, height=height)

    g.ax_joint.plot(xs, ys, color='red', linestyle=':', label='E[S]')
    g.ax_joint.fill_between(xs, np.maximum(ys - stds, 0), np.minimum(ys + stds, 1), alpha=0.2, color='red',
                            label='$E[S] \pm \sigma$')
    g.ax_joint.fill_between(xs, np.maximum(ys - 2 * stds, 0), np.minimum(ys + 2 * stds, 1), alpha=0.08, color='red',
                            label='$E[S] \pm 2\sigma$')
    g.ax_joint.axvline(x=.5, linestyle='--', color='black', lw=1)
    g.ax_joint.axhline(y=.5, linestyle='--', color='black', lw=1)


def plot_result_distribution(plan_df, n_samples=5000, symmetry=True):
    seats, votes = sample_elections(plan_df, n=n_samples, p_seats=True)
    plt.figure(figsize=(10, 10))
    sns.kdeplot(votes, seats, cmap='Reds', shade_lowest=False, shade=True)
    if symmetry:
        ax = sns.kdeplot(1 - votes, 1 - seats, cmap='Blues', shade_lowest=False, shade=True)
    ax.axvline(x=.5, linestyle='--', color='black', lw=1)
    ax.axhline(y=.5, linestyle='--', color='black', lw=1)