import os
import pickle
import random

import dgl

import networkx as nx
import numpy as np
import torch
import utils
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

random.seed(42)


def onehot_encoding_node(m, embedding_dim):
    H = []
    for i in m.nodes:
        H.append(utils.node_feature(m, i, embedding_dim))
    H = np.array(H)
    return H


def sum_var_parts(tsize, lens):
    ind_x = torch.repeat_interleave(torch.arange(lens.size(0)), lens)
    indices = torch.cat(
        [torch.unsqueeze(ind_x, dim=0), torch.unsqueeze(torch.arange(tsize), dim=0)],
        dim=0,
    )
    M = torch.sparse_coo_tensor(
        indices, torch.ones(tsize, dtype=torch.float32), size=[lens.size(0), tsize]
    )
    return M


class BaseDataset(Dataset):
    def __init__(self, keys, data_dir, embedding_dim=20):
        self.keys = keys
        self.data_dir = data_dir
        self.embedding_dim = embedding_dim

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # idx = 0
        key = self.keys[idx]
        with open(os.path.join(self.data_dir, key), "rb") as f:
            data = pickle.load(f)
            if len(data) == 3:
                m1, m2, mapping = data
            else:
                m1, m2 = data
                mapping = []
        
        

        # Prepare subgraph
        n1 = m1.number_of_nodes()
        adj1 = nx.to_numpy_array(m1) + np.eye(n1)
        H1 = onehot_encoding_node(m1, self.embedding_dim)

        # Prepare source graph
        n2 = m2.number_of_nodes()
        adj2 = nx.to_numpy_array(m2) + np.eye(n2)
        H2 = onehot_encoding_node(m2, self.embedding_dim)

        #modified for noniso
        lst = []
        for i, j in enumerate(mapping):
            if i < n1:
                lst.append(j[0])
        non_modified = np.zeros([n1])
        non_modified[lst] = 1.

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

        H1 = np.concatenate([H1, np.zeros((n1, self.embedding_dim))], 1)
        H2 = np.concatenate([np.zeros((n2, self.embedding_dim)), H2], 1)
        H = np.concatenate([H1, H2], 0)

        # prepare graph and cross_graph
        src_lst, dst_lst = np.where(agg_adj1 == 1)
        e = [(i, j) for i, j in zip(src_lst, dst_lst)]
        graph_pt = nx.Graph(e)
        src_lst_cross, dst_lst_cross = np.where(agg_adj2 == 1)
        e_cross = [(i, j) for i, j in zip(src_lst_cross, dst_lst_cross)]
        graph_pt_cross = nx.Graph(e_cross)
        coors = []
        for id in m1.nodes:
            coors.append(m1.nodes[id]["coords"])
        for id in m2.nodes:
            coors.append(m2.nodes[id]["coords"])
        coors = np.vstack(coors)

        # node indice for aggregation
        valid = np.zeros((n1 + n2,))
        valid[:n1] = 1

        # create mapping matrix
        mapping_matrix = np.zeros_like(agg_adj1)
        if len(mapping) > 0:
            mapping = np.array(mapping).T
            mapping[1] = mapping[1] + n1
            mapping_matrix[mapping[0], mapping[1]] = 1.0
            mapping_matrix[mapping[1], mapping[0]] = 1.0

        same_label_matrix = np.zeros_like(agg_adj1)
        same_label_matrix[:n1, n1:] = np.copy(dm_new)
        same_label_matrix[n1:, :n1] = np.copy(np.transpose(dm_new))

        # iso to class
        Y = 1 if "iso" in key else 0

        

        # if n1+n2 > 300 : return None
        sample = {
            "graph": graph_pt,
            "cross_graph": graph_pt_cross,
            "H": H,
            "C": coors,
            "Y": Y,
            "V": valid,
            "mapping": mapping_matrix,
            "same_label": same_label_matrix,
            "non_modified": non_modified,
            "distance_mat": dm_new,
            "key": key,
        }

        return sample


class UnderSampler(Sampler):
    def __init__(self, weights, num_samples, replacement=True):
        weights = np.array(weights) / np.sum(weights)
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        # return iter(torch.multinomial(self.weights, self.num_samples, self.replacement).tolist())
        retval = np.random.choice(
            len(self.weights),
            self.num_samples,
            replace=self.replacement,
            p=self.weights,
        )
        return iter(retval.tolist())

    def __len__(self):
        return self.num_samples


def collate_fn(batch):
    max_natoms = max([len(item["H"]) for item in batch if item is not None])

    M = np.zeros((len(batch), max_natoms, max_natoms))
    S = np.zeros((len(batch), max_natoms, max_natoms))
    Y = np.zeros((len(batch),))

    graph = []
    cross_graph = []
    C = []
    H = []
    V = []
    N1 = []
    keys = []
    nm = []
    dm = []

    for i in range(len(batch)):
        natom = len(batch[i]["H"])

        M[i, :natom, :natom] = batch[i]["mapping"]
        S[i, :natom, :natom] = batch[i]["same_label"]
        Y[i] = batch[i]["Y"]
        graph.append(batch[i]["graph"])
        cross_graph.append(batch[i]["cross_graph"])
        C.append(batch[i]["C"])
        H.append(batch[i]["H"])
        V.append(batch[i]["V"])
        N1.append(np.sum(batch[i]["V"]))
        nm.append(batch[i]["non_modified"])
        dm.append(torch.tensor(batch[i]["distance_mat"]))
        keys.append(batch[i]["key"])

    M = torch.from_numpy(M).float()
    S = torch.from_numpy(S).float()
    Y = torch.from_numpy(Y).float()
    C = torch.from_numpy(np.vstack(C)).float()
    H = torch.from_numpy(np.vstack(H)).float()
    V = torch.from_numpy(np.concatenate(V)).float()
    N1 = torch.tensor(N1, dtype=torch.long)
    nm = torch.from_numpy(np.concatenate(nm)).float()

    return graph, cross_graph, M, S, Y, V, N1, C, H, nm, dm, keys
