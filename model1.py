import random

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers1 import IEGMN_Layer


def def_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sum_var_parts(t, lens):
    chunks = torch.split(t, lens)
    chunks = [torch.sum(i, dim=0) for i in chunks]
    return torch.vstack(chunks)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class gnn(torch.nn.Module):
    def __init__(self, args):
        super(gnn, self).__init__()
        n_FC_layer = args.n_FC_layer
        d_FC_layer = args.d_FC_layer
        self.dropout_rate = args.dropout_rate
        self.branch = args.branch
        self.n_iegmn_layer = args.iegmn_n_lays
        self.input_iegmn_dim = args.residue_emb_dim
        self.iegmn_hid_dim = args.iegmn_lay_hid_dim
        self.num_att_heads = args.num_att_heads
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mse = nn.MSELoss()

        # IEGMN layer
        self.iegmn_layers = nn.ModuleList()
        self.iegmn_layers.append(
            IEGMN_Layer(
                orig_h_feats_dim=self.input_iegmn_dim,
                h_feats_dim=self.input_iegmn_dim,
                out_feats_dim=self.iegmn_hid_dim,
                args=args,
            )
        )

        for i in range(1, self.n_iegmn_layer):
            self.iegmn_layers.append(
                IEGMN_Layer(
                    orig_h_feats_dim=self.input_iegmn_dim,
                    h_feats_dim=self.iegmn_hid_dim,
                    out_feats_dim=self.iegmn_hid_dim,
                    args=args,
                )
            )

        # Fully connected
        self.FC = nn.ModuleList(
            [
                nn.Linear(self.iegmn_hid_dim, d_FC_layer)
                if i == 0
                else nn.Linear(d_FC_layer, 1)
                if i == n_FC_layer - 1
                else nn.Linear(d_FC_layer, d_FC_layer)
                for i in range(n_FC_layer)
            ]
        )

        # embedding graph's feature before feeding them to iegmn_layers (N,40) -> (N,64)
        self.embede = nn.Linear(
            2 * args.embedding_dim, self.input_iegmn_dim, bias=False
        )

        self.theta = torch.tensor(args.al_scale)
        self.zeros = torch.zeros(1)
        if args.ngpu > 0:
            self.theta = self.theta.cuda()
            self.zeros = self.zeros.cuda()

    def embede_graph(self, X):
        graph, cross_graph, c_valid, n1, Y, nm, samelb_mask = X
        X_pt = graph.ndata["coords"]

        # First: Embede the feature of graph
        c_hs = self.embede(graph.ndata["feat"])
        original_coords = graph.ndata["coords"]
        original_feats = c_hs

        attention = None

        # IEGMN layers
        for k in range(len(self.iegmn_layers)):
            if self.branch == "left":
                if k == len(self.iegmn_layers) - 1:
                    X_pt, c_hs1, attention = self.iegmn_layers[k](
                        graph, X_pt, c_hs, original_coords, original_feats, n1, True
                    )
                else:
                    X_pt, c_hs1 = self.iegmn_layers[k](
                        graph, X_pt, c_hs, original_coords, original_feats, n1
                    )
                c_hs1 = -c_hs1
            elif self.branch == "right":
                c_hs1 = 0
            else:
                X_pt, c_hs1 = self.iegmn_layers[k](
                    graph, X_pt, c_hs, original_coords, original_feats, n1
                )

            if self.branch == "left":
                c_hs2 = 0
            else:
                if k == len(self.iegmn_layers) - 1:
                    X_pt, c_hs2, attention = self.iegmn_layers[k](
                        cross_graph,
                        X_pt,
                        c_hs,
                        original_coords,
                        original_feats,
                        n1,
                        True,
                    )
                else:
                    X_pt, c_hs2 = self.iegmn_layers[k](
                        cross_graph, X_pt, c_hs, original_coords, original_feats, n1
                    )

            c_hs = c_hs2 - c_hs1
            c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)

        c_hs = c_hs * c_valid.unsqueeze(-1).repeat(1, c_hs.size(-1))
        c_hs1 = sum_var_parts(c_hs, graph.batch_num_nodes().tolist())
        
        c_hs = c_hs1 / n1.unsqueeze(-1).repeat(1, c_hs.size(-1))
        

        # Update coords node's data for graph and cross graph
        graph.ndata["upd_coords"] = X_pt
        cross_graph.ndata["upd_coords"] = X_pt

        return c_hs, graph, attention

    def fully_connected(self, c_hs):
        # regularization = torch.empty(len(self.FC)*1-1, device=c_hs.device)

        for k in range(len(self.FC)):
            if k < len(self.FC) - 1:
                c_hs = self.FC[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)
                c_hs = F.relu(c_hs)
            else:
                c_hs = self.FC[k](c_hs)

        c_hs = torch.sigmoid(c_hs)

        return c_hs

    def forward(self, X, attn_masking=None, training=False, is_train=True):
        # embede a graph to a vector
        graph, cross_graph, c_valid, n1, Y, nm, samelb_mask = X
        n = cross_graph.batch_num_nodes()

        c_hs, graph, attention = self.embede_graph(X)
        
        # fully connected NN
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1)

        # atten = torch.softmax(attention, dim=1)
        atten = self.cal_atten_batch(n1, n, attention)
        atten = F.normalize(atten)

        attn_loss = self.cal_attn_loss(atten, attn_masking)
        rmsd_loss, centroid_loss, temp = self.cal_rmsd_loss(graph, attention, n1, Y, nm, samelb_mask,is_train)
        pairdst_loss = self.cal_pairdst_loss(graph, n1)

        # note that if you don't use concrete dropout, regularization 1-2 is zero
        if training:
            return c_hs, attn_loss, rmsd_loss, pairdst_loss, centroid_loss, temp
        else:
            return c_hs

    def cal_attn_loss(self, attention, attn_masking):
        mapping, samelb = attn_masking

        top = torch.exp(-(attention * mapping))
        top = torch.where(mapping == 1.0, top, self.zeros)
        top = top.sum((1, 2))

        topabot = torch.exp(-(attention * samelb))
        topabot = torch.where(samelb == 1.0, topabot, self.zeros)
        topabot = topabot.sum((1, 2))

        return (top / (topabot - top + 1)).sum(0) * self.theta / attention.shape[0]

    def cal_rmsd_loss(self, batch_graph, attention, n1, Y, non_modified, samelb_mask, is_train):
        a = torch.cumsum(n1, dim=0).tolist()
        a.insert(0, 0)

        # batch_rmsd_loss = []
        batch_rmsd_loss = torch.zeros([]).to(self.device) 
        centroid_loss = torch.zeros([]).to(self.device)

        PP, QQ = self.get_coords(batch_graph, n1)
        # attention =  torch.softmax(attention, dim=1)
        
        # attention = samelb_mask*attention

        index = attention.max(1, keepdim=True)[1]
        mapping = torch.zeros_like(attention).scatter_(1, index, 1.0)
        mapping = mapping - attention.detach() + attention

        # attention = F.normalize(attention)
        #mapping = F.gumbel_softmax(attention, tau=1, hard=True)
        temp = []
        QQ = torch.mm(mapping, QQ)
        for i in range(len(a) - 1):
            P = PP[a[i] : a[i + 1], :]
            Q = QQ[a[i] : a[i + 1], :]
            nm = non_modified[a[i] : a[i + 1]]
            P_mean = P.mean(dim=0)
            Q_mean = Q.mean(dim=0)
            h = (P - P_mean).transpose(0,1) @ (Q - Q_mean)/float(self.num_att_heads)
            u, S, vt = torch.linalg.svd(h)
            num_it = 0
            while (
                torch.min(S) < 1e-3
                or torch.min(
                    torch.abs(
                        (S**2).view(1, 3)
                        - (S**2).view(3, 1)
                        + torch.eye(3).to(self.device)
                    )
                )
                < 1e-2
            ):
                h = h + torch.rand(3, 3).to(self.device) * torch.eye(3).to(self.device)
                u, S, vt = torch.linalg.svd(h)
                num_it += 1
                if num_it > 10:
                    print("unstable\n")
                    break

            corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(h))], device=self.device))
            r = (u @ corr_mat) @ vt

            trans = Q_mean - torch.t(r @ P_mean.t())  # (1,3)

            P_predict = (r @ P.T).T + trans
            # rmsd = torch.sqrt(torch.mean(torch.sum((P_predict - Q) ** 2, axis=1)))
            rmsd = self.mse(P_predict, Q)
            # ms = rmsd*Y[i] + (1-Y[i])/(rmsd+1e-8)
            if is_train:
                idx = torch.where(nm == 1.)
                if len(idx[0]) <1:
                    new_rmsd = 0.
                else:
                    new_rmsd = self.mse(P_predict[idx], Q[idx])
            else:
                new_rmsd = self.mse(P_predict, Q)
            # new_rmsd = self.mse(P_predict, Q) * Y[i]
            
            P_predict_mean = P_predict.mean(dim=0)
            cen = self.mse(P_predict_mean, Q_mean)
            # ms = rmsd*Y[i] + 0.0001*(1-Y[i])/(rmsd+1e-8) 
            # ms = rmsd*Y[i] + 0.0005*(1-Y[i])*rmsd 

            batch_rmsd_loss = batch_rmsd_loss + new_rmsd
            centroid_loss = centroid_loss + cen 
            temp.append(rmsd.item())
        batch_rmsd_loss = batch_rmsd_loss / float(n1.shape[0])
        centroid_loss = centroid_loss / float(n1.shape[0])
        print(temp)
        # return torch.tensor(batch_rmsd_loss, device=attention.device)
        return batch_rmsd_loss, centroid_loss, temp 

    def get_coords(self, batch_graph, n1):
        sub_coords = []
        graph_coords = []
        bg_list = dgl.unbatch(batch_graph)
        for i, g in enumerate(bg_list):
            sub_coords.append(g.ndata["upd_coords"][: n1[i]])
            graph_coords.append(g.ndata["coords"][n1[i] :])

        return torch.vstack(sub_coords), torch.vstack(graph_coords)

    def calculate_nodes_dst(self, edges):
        pdist = nn.PairwiseDistance(p=2)
        return {"dst": pdist(edges.src["coords"], edges.dst["coords"])}
    def calculate_updnodes_dst(self, edges):
        pdist = nn.PairwiseDistance(p=2)
        return {"upd_dst": pdist(edges.src["upd_coords"], edges.dst["upd_coords"])}

    def cal_pairdst_loss(self, batch_graph, n1):
        batch_lst = dgl.unbatch(batch_graph)
        batch_pairwise_loss = torch.zeros([]).to(self.device)
        for i, g in enumerate(batch_lst):
            lst_nodes = torch.arange(n1[i]).tolist()
            g1 = dgl.node_subgraph(g, lst_nodes)

            g1.apply_edges(self.calculate_nodes_dst)
            g1.apply_edges(self.calculate_updnodes_dst)
            dst_loss = (g1.edata["upd_dst"] - g1.edata["dst"]).sum().abs()
            batch_pairwise_loss = batch_pairwise_loss + dst_loss
        batch_pairwise_loss = batch_pairwise_loss / float(n1.shape[0])
        return batch_pairwise_loss

    def get_refined_adjs2(self, X):
        _, attention = self.embede_graph(X)
        return attention

    def cal_atten_batch(self, n1, n, attention):
        n2 = n - n1
        atten_batch = torch.zeros((len(n1), max(n), max(n))).to(self.device)
        i = torch.cumsum(n1, dim=0).tolist()
        i.insert(0, 0)
        j = torch.cumsum(n2, dim=0).tolist()
        j.insert(0, 0)
        for k in range(len(n1)):
            atten_batch[k][: n1[k], n1[k] : n[k]] = attention[
                i[k] : i[k + 1], j[k] : j[k + 1]
            ]
            atten_batch[k][n1[k] : n[k], : n1[k]] = attention[
                i[k] : i[k + 1], j[k] : j[k + 1]
            ].transpose(0, 1)
        return atten_batch

