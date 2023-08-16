import math
import dgl
import torch
from torch import nn
from dgl import function as fn
import os
import pickle
import random
from collections import defaultdict

import utils
from kabsch import kabsch_rmsd

import networkx as nx
import numpy as np
import torch
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

def onehot_encoding_node(m, embedding_dim):
    H = []
    for i in m.nodes:
        H.append(utils.node_feature(m, i, embedding_dim))
    H = np.array(H)
    return H

data_path = os.path.join("data_processed", "tiny")
train_keys = os.path.join(data_path, "train_keys.pkl")
with open(train_keys, "rb") as fp:
    train_keys = pickle.load(fp)
with open(os.path.join(data_path, train_keys[0]), "rb") as f:
    data = pickle.load(f)
    if len(data) == 3:
        m1, m2, mapping = data
    else:
        m1, m2 = data
        mapping = []

        
n1 = m1.number_of_nodes()
adj1 = nx.to_numpy_array(m1) + np.eye(n1)
H1 = onehot_encoding_node(m1, 20)

# Prepare source graph
n2 = m2.number_of_nodes()
adj2 = nx.to_numpy_array(m2) + np.eye(n2)
H2 = onehot_encoding_node(m2, 20)

# Aggregation node encoding
agg_adj1 = np.zeros((n1 + n2, n1 + n2))
agg_adj1[:n1, :n1] = adj1
agg_adj1[n1:, n1:] = adj2
agg_adj2 = np.copy(agg_adj1)
dm = distance_matrix(H1, H2)
dm_new = np.zeros_like(dm)
dm_new[dm == 0.0] = 1.0
agg_adj2[:n1, n1:] = np.copy(dm_new)
agg_adj2[n1:, :n1] = np.copy(np.transpose(dm_new))

H1 = np.concatenate([H1, np.zeros((n1, 20))], 1)
H2 = np.concatenate([np.zeros((n2, 20)), H2], 1)
H = np.concatenate([H1, H2], 0)

#prepare graph and cross_graph
src_lst, dst_lst = np.where(agg_adj1==1)
graph_pt = dgl.graph((src_lst, dst_lst))
src_lst_cross, dst_lst_cross = np.where(agg_adj2==1)
graph_pt_cross = dgl.graph((src_lst_cross, dst_lst_cross))
graph_pt.ndata['feat'] = torch.from_numpy(H).float()
graph_pt_cross.ndata['feat'] = torch.from_numpy(H).float()
X_pt = []
for id in m1.nodes:
    X_pt.append(m1.nodes[id]["coords"])
for id in m2.nodes:
    X_pt.append(m2.nodes[id]["coords"])
X_pt = np.vstack(X_pt)
X_pt = torch.from_numpy(X_pt).float()
graph_pt.ndata['coords'] = X_pt
graph_pt_cross.ndata['coords'] = X_pt
device = torch.device("cuda")
bg = dgl.batch([graph_pt, graph_pt_cross])
bg = bg.to(device)