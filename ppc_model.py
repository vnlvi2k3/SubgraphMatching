import torch
import torch.nn as nn
import torch.nn.functional as F
from ppc_layers import IEGMN_Layer

from layers import GAT_gate


class gnn(torch.nn.Module):
    def __init__(self, args):
        super(gnn, self).__init__()
        n_graph_layer = args.n_graph_layer
        d_graph_layer = args.d_graph_layer
        n_FC_layer = args.n_FC_layer
        d_FC_layer = args.d_FC_layer
        self.dropout_rate = args.dropout_rate
        self.branch = args.branch
        self.n_iegmn_layer = args.iegmn_n_lays
        self.input_iegmn_dim = args.residue_emb_dim
        self.iegmn_hid_dim = args.iegmn_lay_hid_dim

        self.iegmn_layers = nn.ModuleList()
        self.iegmn_layers.append(
            IEGMN_Layer(orig_h_feats_dim=self.input_iegmn_dim,
                       h_feats_dim=self.input_iegmn_dim,
                       out_feats_dim=self.iegmn_hid_dim,
                       args=args))
        
        for i in range(1, self.n_iegmn_layer):
            self.iegmn_layers.append(
                IEGMN_Layer(orig_h_feats_dim=self.input_iegmn_dim,
                           h_feats_dim=self.iegmn_hid_dim,
                           out_feats_dim=self.iegmn_hid_dim,
                           args=args))

        self.FC = nn.ModuleList(
            [
                nn.Linear(self.layers1[-1], d_FC_layer)
                if i == 0
                else nn.Linear(d_FC_layer, 1)
                if i == n_FC_layer - 1
                else nn.Linear(d_FC_layer, d_FC_layer)
                for i in range(n_FC_layer)
            ]
        )

        self.embede = nn.Linear(2 * args.embedding_dim,
                                self.input_iegmn_dim, bias=False)
        self.theta = torch.tensor(args.al_scale)
        self.zeros = torch.zeros(1)
        if args.ngpu > 0:
            self.theta = self.theta.cuda()
            self.zeros = self.zeros.cuda()

    def embede_graph(self, X):
        graph, cross_graph, c_valid = X

        X_pt = graph.ndata['coords']
        c_hs = self.embede(graph.ndata['feat'])
        original_coords = graph.ndata["coords"]
        original_feats = c_hs

        attention = None
        #graph, coords, h_feats, original_node_features, original_coords

        for k in range(len(self.iegmn_layers)):
            if self.branch == "left":
                if k == len(self.iegmn_layers) - 1:
                    X_pt, c_hs1, attention = self.iegmn_layers[k](graph, X_pt, c_hs, original_coords, original_feats, True)
                else:
                    X_pt, c_hs1 = self.iegmn_layers[k](graph, X_pt, c_hs, original_coords, original_feats)
                c_hs1 = - c_hs1
            elif self.branch == "right":
                c_hs1 = 0
            else:
                X_pt, c_hs1 = self.iegmn_layers[k](graph, X_pt, c_hs, original_coords, original_feats)

            if self.branch == "left":
                c_hs2 = 0
            else:
                if k == len(self.iegmn_layers) - 1:
                    X_pt, c_hs2, attention = self.iegmn_layers[k](cross_graph, X_pt, c_hs, original_coords, original_feats, True)
                else:
                    X_pt, c_hs2 = self.iegmn_layers[k](cross_graph, X_pt, c_hs, original_coords, original_feats)

            c_hs = c_hs2 - c_hs1
            c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)

        c_hs = c_hs * c_valid.unsqueeze(-1).repeat(1, 1, c_hs.size(-1))
        c_hs = c_hs.sum(1) / c_valid.sum(1, keepdim=True)
        return c_hs, X_pt, F.normalize(attention)

    def fully_connected(self, c_hs):
        # regularization = torch.empty(len(self.FC)*1-1, device=c_hs.device)

        for k in range(len(self.FC)):
            if k < len(self.FC) - 1:
                c_hs = self.FC[k](c_hs)
                c_hs = F.dropout(c_hs, p=self.dropout_rate,
                                 training=self.training)
                c_hs = F.relu(c_hs)
            else:
                c_hs = self.FC[k](c_hs)

        c_hs = torch.sigmoid(c_hs)

        return c_hs

    def forward(self, X, attn_masking=None, training=False):
        # embede a graph to a vector
        c_hs, X_pt, attention = self.embede_graph(X)

        # fully connected NN
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1)

        # note that if you don't use concrete dropout, regularization 1-2 is zero
        if training:
            return c_hs, X_pt, self.cal_attn_loss(attention, attn_masking)
        else:
            return c_hs, X_pt

    def cal_attn_loss(self, attention, attn_masking):
        mapping, samelb = attn_masking

        top = torch.exp(-(attention * mapping))
        top = torch.where(mapping == 1.0, top, self.zeros)
        top = top.sum((1, 2))

        topabot = torch.exp(-(attention * samelb))
        topabot = torch.where(samelb == 1.0, topabot, self.zeros)
        topabot = topabot.sum((1, 2))

        return (top / (topabot - top + 1)).sum(0) * self.theta / attention.shape[0]

    def get_refined_adjs2(self, X):
        _, attention = self.embede_graph(X)
        return attention
