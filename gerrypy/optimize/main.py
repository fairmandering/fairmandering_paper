import networkx as nx
import pandas as pd
import pickle
import numpy as np
from gerrypy.optimize import partition
from gerrypy.optimize.cost import expected_rep_gap
from gerrypy.data.synthetic import generate_synthetic_input
from gerrypy.optimize.prune import make_lengths_data, complete_lengths_data
from gerrypy.optimize.problems.master import make_master
from gerrypy.optimize.tree import SampleTree
from gurobipy import *
import gpytorch
from gerrypy.gp import exact
from gerrypy.optimize import annotate
from gerrypy.analyze.districts import *
from scipy.stats import norm


def load_real_data(data_base_path):
    state_df_path = os.path.join(data_base_path, 'state_df.csv')
    adjacency_graph_path = os.path.join(data_base_path, 'G.p')

    state_df = pd.read_csv(state_df_path)
    G = nx.read_gpickle(adjacency_graph_path)

    if os.path.exists(os.path.join(data_base_path, 'lengths.npy')):
        lengths_path = os.path.join(data_base_path, 'lengths.npy')
        lengths = np.load(lengths_path)
    else:
        from scipy.spatial.distance import pdist, squareform
        lengths = squareform(pdist(state_df[['x', 'y']].values))

    if os.path.exists(os.path.join(data_base_path, 'edge_dists.p')):
        edge_dists_path = os.path.join(data_base_path, 'edge_dists.p')
        edge_dists = pickle.load(open(edge_dists_path, 'rb'))
    else:
        edge_dists = dict(nx.all_pairs_shortest_path_length(G))

    return state_df, G, lengths, edge_dists



