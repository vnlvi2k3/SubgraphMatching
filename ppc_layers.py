import math

import dgl
import networkx as nx

import torch
from torch import nn
from dgl import function as fn

class GraphNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, is_node=True):
        super().__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine
        self.is_node = is_node

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(self.num_features))
            self.beta = nn.Parameter(torch.zeros(self.num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def norm(self, x):
        mean = x.mean(dim=0, keepdim=True)
        var = x.std(dim=0, keepdim=True)
        return (x - mean) / (var + self.eps)

    def forward(self, g, h):
        graph_size = g.batch_num_nodes() if self.is_node else g.batch_num_edges()
        x_list = torch.split(h, graph_size.tolist())
        norm_list = []
        for x in x_list:
            norm_list.append(self.norm(x))
        norm_x = torch.cat(norm_list, 0)

        if self.affine:
            return self.gamma * norm_x + self.beta
        else:
            return norm_x

def get_mask(ligand_batch_num_nodes, receptor_batch_num_nodes, device):
    rows = sum(ligand_batch_num_nodes)
    cols = sum(receptor_batch_num_nodes)
    mask = torch.zeros(rows, cols).to(device)
    partial_l = 0
    partial_r = 0
    for l_n, r_n in zip(ligand_batch_num_nodes, receptor_batch_num_nodes):
        mask[partial_l: partial_l + l_n, partial_r: partial_r + r_n] = 1
        partial_l = partial_l + l_n
        partial_r = partial_r + r_n
    return mask


def get_non_lin(type, negative_slope):
    if type == 'swish':
        return nn.SiLU()
    else:
        assert type == 'lkyrelu'
        return nn.LeakyReLU(negative_slope=negative_slope)


def get_layer_norm(layer_norm_type, dim):
    if layer_norm_type == 'BN':
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == 'LN':
        return nn.LayerNorm(dim)
    else:
        return nn.Identity()


def get_final_h_layer_norm(layer_norm_type, dim):
    if layer_norm_type == 'BN':
        return nn.BatchNorm1d(dim)
    elif layer_norm_type == 'LN':
        return nn.LayerNorm(dim)
    elif layer_norm_type == 'GN':
        return GraphNorm(dim)
    else:
        assert layer_norm_type == '0'
        return nn.Identity()


def apply_final_h_layer_norm(g, h, norm_type, norm_layer):
    if norm_type == 'GN':
        return norm_layer(g, h)
    return norm_layer(h)


def compute_cross_attention(queries, keys, values, mask, cross_msgs):
    if not cross_msgs:
        return queries * 0.
    a = mask * torch.mm(queries, torch.transpose(keys, 1, 0)) - 1000. * (1. - mask)
    a_x = torch.softmax(a, dim=1)  # i->j, NxM, a_x.sum(dim=1) = torch.ones(N)
    attention_x = torch.mm(a_x, values)  # (N,d)
    return a_x, attention_x

def get_features(batch_graph, n1):
    sub_feats= []
    graph_feats = []
    bg_list = dgl.unbatch(batch_graph)
    for i, g in enumerate(bg_list):
        sub_feats.append(g.ndata["feat"][:n1[i]])
        graph_feats.append(g.ndata["feat"][n1[i]:])

    sub_feats = torch.vstack(sub_feats)
    graph_feats = torch.vstack(graph_feats)
    return sub_feats, graph_feats

class IEGMN_Layer(nn.Module):
    def __init__(self, orig_h_feats_dim, h_feats_dim, out_feats_dim, args):
        super(IEGMN_Layer, self).__init__()
        dropout = args.dropout_rate
        nonlin = args.nonlin
        self.cross_msgs = args.cross_msgs
        layer_norm = args.layer_norm
        layer_norm_coors = args.layer_norm_coors
        self.final_h_layer_norm = args.final_h_layer_norm
        self.use_dist_in_layers = args.use_dist_in_layers
        self.skip_weight_h = args.skip_weight_h
        self.x_connection_init = args.x_connection_init
        leakyrelu_neg_slope = args.leakyrelu_neg_slope
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        
        self.h_feats_dim = h_feats_dim
        self.out_feats_dim = out_feats_dim
        self.all_sigmas_dist = [1.5 ** x for x in range(15)]
        
        edge_mlp_input_dim = (h_feats_dim * 2) + len(self.all_sigmas_dist)
            
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_mlp_input_dim, self.out_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            get_layer_norm(layer_norm, self.out_feats_dim),
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
        )
        
        self.node_norm = nn.Identity()
        self.att_mlp_Q = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_K = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
            get_non_lin(nonlin, leakyrelu_neg_slope),
        )
        self.att_mlp_V = nn.Sequential(
            nn.Linear(h_feats_dim, h_feats_dim, bias=False),
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(orig_h_feats_dim + 2 * h_feats_dim + self.out_feats_dim, h_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            get_layer_norm(layer_norm, h_feats_dim),
            nn.Linear(h_feats_dim, out_feats_dim),
        )
        
        self.final_h_layernorm_layer = get_final_h_layer_norm(self.final_h_layer_norm, out_feats_dim)
        
        self.coors_mlp = nn.Sequential(
            nn.Linear(self.out_feats_dim, self.out_feats_dim),
            nn.Dropout(dropout),
            get_non_lin(nonlin, leakyrelu_neg_slope),
            get_layer_norm(layer_norm_coors, self.out_feats_dim),
            nn.Linear(self.out_feats_dim, 1)
        )
                    
    def apply_edges1(self, edges):
        return {'cat_feat': torch.cat([edges.src['feat'], edges.dst['feat']], dim=1)}
    

                
    def forward(self, graph, coords, h_feats, original_node_features, original_coords,
                 bacth_sub_numnode,  get_attention=False):
        with graph.local_scope():
            #X_i, H_i : current node's coords and features
            graph.ndata['x_now'] = coords
            graph.ndata['feat'] = h_feats  
            
            #Calculate X_rel = |x_i - x_j| ..... with (i,j) is edges of graph
            graph.apply_edges(fn.u_sub_v('x_now', 'x_now', 'x_rel')) 
            
            #X_rel_mag =  exp(-||x_i - x_j||^2 / sigma) ..... sigma = 1.5 ** x for x = [0, 15]
            #len(X_rel_mag = 15)
            x_rel_mag = graph.edata['x_rel']**2
            x_rel_mag = torch.sum(x_rel_mag, dim=1, keepdim=True)
            x_rel_mag = torch.cat([torch.exp(-x_rel_mag / sigma) for sigma in self.all_sigmas_dist], dim=-1)

            if not self.use_dist_in_layers:
                x_rel_mag = x_rel_mag * 0.
                
            #cat_input_for_msg = (h_i, h_j, x_rel_mag)
            graph.apply_edges(self.apply_edges1)
            cat_input_for_msg = torch.cat((graph.edata['cat_feat'], x_rel_mag), dim=-1)
            
            #msg = mlp_edge(h_i, h_j, x_rel_mag)
            graph.edata['msg'] = self.edge_mlp(cat_input_for_msg)
            
            #Calculate cross attention: 
            #attention (N,M): attention of subgraph to graph     
            #aggr_cross1 (N,d) =  attention (N,M) * value_subgraph (M,d)
            #aggr_cross2 (M,d) =  attention.transpose() * value_subgraph (N,d)
            #aggr_cross : concatenate aggr_cross1 and aggr_cross2 as a node data for big graph
            #aggr_cross_msg(i) = sum_j a_{i,j} * h_j
            mask = get_mask(bacth_sub_numnode, graph.batch_num_nodes() - bacth_sub_numnode, self.device)
            sub_feats, graph_feats = get_features(graph, bacth_sub_numnode)
            attention, aggr_cross1 = compute_cross_attention(self.att_mlp_Q(sub_feats), self.att_mlp_K(graph_feats)
                                                   , self.att_mlp_V(graph_feats), mask, self.cross_msgs) #(n x m), (n x d)
            
            _, aggr_cross2 = compute_cross_attention(self.att_mlp_Q(graph_feats), self.att_mlp_K(sub_feats)
                                                   , self.att_mlp_V(sub_feats), mask.transpose(0,1), self.cross_msgs) #(m x n), (m x d)
            
            graph.ndata['aggr_cross_msg'] = torch.cat((aggr_cross1, aggr_cross2), dim=0)

            #edge_coef : phi^x(m_{i->j})
            edge_coef = self.coors_mlp(graph.edata['msg'])
            #data[x_moment] = (x_i - x_j) * phi^x(m_{i->j})
            graph.edata['x_moment'] = graph.edata['x_rel']*edge_coef
            
            #data[aggr_msg]: sum_j m_{i->j}
            #data[x_update] : sum_j (x_i - x_j) * phi^x(m_{i->j}) : X_update coordinate
            graph.update_all(fn.copy_e('x_moment', 'm'), fn.mean('m', 'x_update'))
            graph.update_all(fn.copy_e('msg', 'm'), fn.mean('m', 'aggr_msg'))
            
            print("x connection:\n", self.x_connection_init.shape)
            print("orig:\n", original_coords.shape)
            print("x now:\n", graph.ndata['x_now'].shape)
            print("x update:\n", graph.ndata['x_update'].shape)
            
            x_final = self.x_connection_init * original_coords + \
                            (1.-self.x_connection_init)*graph.ndata['x_now'] +\
                            graph.ndata['x_update']
            
            input_node_upd = torch.cat((self.node_norm(graph.ndata['feat']),
                                                graph.ndata['aggr_msg'],
                                                graph.ndata['aggr_cross_msg'],
                                                original_node_features),
                                                dim=-1)
            
            #calculate node update feature
            if self.h_feats_dim == self.out_feats_dim:
                node_upd = self.skip_weight_h * self.node_mlp(input_node_upd) + \
                                (1.-self.skip_weight_h) * h_feats
            else:
                node_upd = self.node_mlp(input_node_upd)
                
            node_upd = apply_final_h_layer_norm(graph, node_upd, self.final_h_layer_norm,
                                                        self.final_h_layernorm_layer)
            
            if get_attention: 
                return x_final, node_upd, attention 
            return x_final, node_upd
