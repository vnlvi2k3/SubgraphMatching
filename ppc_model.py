import torch
import torch.nn as nn
import torch.nn.functional as F
from ppc_layers import IEGMN_Layer
import dgl

def sum_var_parts(t, lens):
    t_size_0 = t.size(0)
    ind_x = torch.repeat_interleave(torch.arange(lens.size(0)).to(lens.device), lens)
    indices = torch.cat(
        [
            torch.unsqueeze(ind_x, dim=0),
            torch.unsqueeze(torch.arange(t_size_0).to(lens.device), dim=0)
        ],
        dim=0
    )
    M = torch.sparse_coo_tensor(
        indices,
        torch.ones(t_size_0, dtype=torch.float32),
        size=[lens.size(0), t_size_0],
        device = t.device
    )
    return M @ t

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
        graph, cross_graph, c_valid, n1 = X
        X_pt = graph.ndata['coords']

        #First: Embede the feature of graph
        c_hs = self.embede(graph.ndata['feat'])
        original_coords = graph.ndata["coords"]
        original_feats = c_hs


        attention = None

        #IEGMN layers
        for k in range(len(self.iegmn_layers)):
            if self.branch == "left":
                if k == len(self.iegmn_layers) - 1:
                    X_pt, c_hs1, attention = self.iegmn_layers[k](graph, X_pt, c_hs, original_coords, original_feats, n1, True)
                else:
                    X_pt, c_hs1 = self.iegmn_layers[k](graph, X_pt, c_hs, original_coords, original_feats, n1)
                c_hs1 = - c_hs1
            elif self.branch == "right":
                c_hs1 = 0
            else:
                X_pt, c_hs1 = self.iegmn_layers[k](graph, X_pt, c_hs, original_coords, original_feats, n1)

            if self.branch == "left":
                c_hs2 = 0
            else:
                if k == len(self.iegmn_layers) - 1:
                    X_pt, c_hs2, attention = self.iegmn_layers[k](cross_graph, X_pt, c_hs, original_coords, original_feats, n1, True)
                else:
                    X_pt, c_hs2 = self.iegmn_layers[k](cross_graph, X_pt, c_hs, original_coords, original_feats, n1)

            c_hs = c_hs2 - c_hs1
            c_hs = F.dropout(c_hs, p=self.dropout_rate, training=self.training)

        c_hs = c_hs * c_valid.unsqueeze(-1).repeat(1, c_hs.size(-1))
        c_hs = sum_var_parts(c_hs, graph.batch_num_nodes())
        c_hs = c_hs / n1.unsqueeze(-1).repeat(1, c_hs.size(-1))

        #Update coords node's data for graph and cross graph
        # graph.ndata["upd_coords"] = X_pt
        # cross_graph.ndata["upd_coords"] = X_pt

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
        graph, cross_graph, c_valid, n1 = X
        n = cross_graph.batch_num_nodes()
        c_hs, graph, attention = self.embede_graph(X)

        # fully connected NN
        c_hs = self.fully_connected(c_hs)
        c_hs = c_hs.view(-1)
        
        attn_loss = self.cal_attn_loss(self.cal_atten_batch2(n1, n, attention), attn_masking)
        rmsd_loss = self.cal_rmsd_loss(c_hs, graph, attention, n1, n)
        # pairdst_loss = self.cal_pairdst_loss(graph)

        # note that if you don't use concrete dropout, regularization 1-2 is zero
        if training:
            return c_hs, attn_loss, rmsd_loss
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
    
    def cal_rmsd_loss(self, pred ,batch_graph, attention, n1, n):
        n2 = n - n1

        a = torch.cumsum(n1, dim=0).tolist()
        a.insert(0,0)
        
        # prob = torch.round(pred)
        batch_lst = dgl.unbatch(batch_graph)
        batch_rmsd_loss = torch.zeros([]).to(self.device)  
        PP, QQ = self.get_coords(batch_graph, n1)
        
        index = attention.max(1, keepdim=True)[1]
        mapping = torch.zeros_like(attention).scatter_(1, index, 1.0)
        
        # mapping = gumbel_softmax(attention, tau=1, hard=True)
        QQ = torch.mm(mapping, QQ)
        for i in range(len(a)-1):
            P = PP[a[i]:a[i+1],:]
            Q = QQ[a[i]:a[i+1],:]
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
            rmsd = torch.sqrt(torch.mean(torch.sum((P_predict - Q) ** 2, axis=1)))
            rmsd = rmsd*prob[i]
            batch_rmsd_loss = batch_rmsd_loss + rmsd
        batch_rmsd_loss = batch_rmsd_loss / (float(len(batch_lst))**2)
        return batch_rmsd_loss

    def get_coords(self, batch_graph, n1):
        sub_coords = []
        graph_coords = []
        bg_list = dgl.unbatch(batch_graph)
        for i, g in enumerate(bg_list):
            sub_coords.append(g.ndata["coords"][:n1[i]])
            graph_coords.append(g.ndata["coords"][n1[i]:])
    
        sub_coords = torch.vstack(sub_coords)
        graph_coords = torch.vstack(graph_coords)
        return sub_coords, graph_coords
    
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
        batch_pairwise_loss = batch_pairwise_loss / float(len(batch_lst))
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
