import torch
import torch.nn as nn
import torch.nn.functional as F
from ppc_layers import IEGMN_Layer
import dgl


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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #IEGMN layer
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

        #Fully connected
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

        #embedding graph's feature before feeding them to iegmn_layers (N,40) -> (N,64)
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

        #First: Embede the feature of graph
        c_hs = self.embede(graph.ndata['feat'])
        original_coords = graph.ndata["coords"]
        original_feats = c_hs

        batch_sub_numnode = torch.sum(c_valid, axis=1).long()

        attention = None

        #IEGMN layers
        for k in range(len(self.iegmn_layers)):
            if self.branch == "left":
                if k == len(self.iegmn_layers) - 1:
                    X_pt, c_hs1, attention = self.iegmn_layers[k](graph, X_pt, c_hs, original_coords, original_feats, batch_sub_numnode, True)
                else:
                    X_pt, c_hs1 = self.iegmn_layers[k](graph, X_pt, c_hs, original_coords, original_feats, batch_sub_numnode)
                c_hs1 = - c_hs1
            elif self.branch == "right":
                c_hs1 = 0
            else:
                X_pt, c_hs1 = self.iegmn_layers[k](graph, X_pt, c_hs, original_coords, original_feats, batch_sub_numnode)

            if self.branch == "left":
                c_hs2 = 0
            else:
                if k == len(self.iegmn_layers) - 1:
                    X_pt, c_hs2, attention = self.iegmn_layers[k](cross_graph, X_pt, c_hs, original_coords, original_feats, batch_sub_numnode, True)
                else:
                    X_pt, c_hs2 = self.iegmn_layers[k](cross_graph, X_pt, c_hs, original_coords, original_feats, batch_sub_numnode)

            c_hs = c_hs2 - c_hs1
            c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)

        c_hs = c_hs * c_valid.unsqueeze(-1).repeat(1, 1, c_hs.size(-1))
        c_hs = c_hs.sum(1) / c_valid.sum(1, keepdim=True)

        #Update coords node's data for graph and cross graph
        graph.ndata["upd_coords"] = X_pt
        cross_graph.ndata["upd_coords"] = X_pt

        return c_hs, graph, F.normalize(attention)

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
        graph, cross_graph, c_valid = X
        n1 = torch.sum(c_valid, axis=1).long()
        n = cross_graph.batch_num_nodes()
        c_hs, graph, attention = self.embede_graph(X)

        # fully connected NN
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1)

        # note that if you don't use concrete dropout, regularization 1-2 is zero
        if training:
            return c_hs, self.cal_attn_loss(self.cal_atten_batch2(attention, n1, n), attn_masking), self.cal_rmsd_loss(c_hs, graph, self.cal_atten_batch1(n1, n, attention), n1, n), self.cal_pairdst_loss(graph)
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
    
    def cal_rmsd_loss(self, pred, batch_graph, attention, n1, n):
        n2 = n - n1
        prob = torch.round(pred)
        batch_lst = dgl.unbatch(batch_graph)
        batch_rmsd_loss = torch.zeros([]).to(self.device)  
        mapping = F.gumbel_softmax(attention, tau=1, hard=True)
        for i, g in enumerate(batch_lst):
            P = g.ndata['upd_coords'][:n1[i],:]
            Q = g.ndata['upd_coords'][n1[i]:,:]
            Q = torch.mm(mapping[i][:n1[i],:n2[i]], Q)
            P_mean = P.mean(dim=0)
            Q_mean = Q.mean(dim=0)
            h = (P - P_mean).T@(Q - Q_mean)
            u, s, vt = torch.linalg.svd(h)
            v = vt.T
            d = torch.sign(torch.det(v@u.T))
            e = torch.tensor([[1,0,0],[0,1,0],[0,0,d]]).to(self.device)
            r = v@e@u.T
            tt = Q_mean - r@P_mean
            P_predict = (r@P.T).T + tt
            rmsd = torch.sqrt(((P_predict - Q)**2).sum() / float(Q.shape[0]))
            rmsd = rmsd*prob[i]
            batch_rmsd_loss = batch_rmsd_loss + rmsd
        batch_rmsd_loss = batch_rmsd_loss / float(len(n1))
        return batch_rmsd_loss
    
    def calculate_nodes_dst(self, edges):
        pdist = nn.PairwiseDistance(p=2)
        return {"dst": pdist(edges.src["coords"], edges.dst["coords"])}
    def calculate_updnodes_dst(self, edges):
        pdist = nn.PairwiseDistance(p=2)
        return {"upd_dst": pdist(edges.src["upd_coords"], edges.dst["upd_coords"])}

    def cal_pairdst_loss(self, batch_graph):
        batch_lst = dgl.unbatch(batch_graph)
        batch_pairwise_loss = torch.zeros([]).to(self.device) 
        for i, g in enumerate(batch_lst):
            g.apply_edges(self.calculate_nodes_dst)
            g.apply_edges(self.calculate_updnodes_dst)
            dst_loss = (g.edata["upd_dst"] - g.edata["dst"]).sum()
            batch_pairwise_loss = batch_pairwise_loss + dst_loss
        batch_pairwise_loss = batch_pairwise_loss / float(len(batch_graph))
        return batch_pairwise_loss 

    def get_refined_adjs2(self, X):
        _, attention = self.embede_graph(X)
        return attention
    
    def cal_atten_batch1(self, n1, n, attention):
        n2 = n - n1
        atten_batch = torch.zeros((len(n1), max(n1), max(n2))).to(self.device)
        i = torch.cumsum(n1, dim=0).tolist()
        i.insert(0,0)
        j = torch.cumsum(n2, dim=0).tolist()
        j.insert(0,0)
        for k in range(len(n1)):
            atten_batch[k][:i[k+1]-i[k], : j[k+1]-j[k]] = attention[i[k]:i[k+1],j[k]:j[k+1]] 
        return atten_batch
    
    def cal_atten_batch2(self, n1, n, attention):
        n2 = n - n1
        atten_batch = torch.zeros((len(n1), max(n), max(n))).to(self.device)
        i = torch.cumsum(n1, dim=0).tolist()
        i.insert(0,0)
        j = torch.cumsum(n2, dim=0).tolist()
        j.insert(0,0)
        for k in range(len(n1)):
            atten_batch[k][:n1[k], n1[k]:n[k]] = attention[i[k]:i[k+1],j[k]:j[k+1]] 
            atten_batch[k][n1[k]:n[k], :n1[k]] = attention[i[k]:i[k+1],j[k]:j[k+1]].transpose(0,1) 
        return atten_batch