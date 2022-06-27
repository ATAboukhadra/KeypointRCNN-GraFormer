from __future__ import absolute_import

import torch.nn as nn
import torch
import numpy as np
import scipy.sparse as sp
import copy, math
import torch.nn.functional as F
import scipy
from torch.nn.parameter import Parameter
from .ChebConv import ChebConv, _ResChebGC
from .GraFormer import GraphNet, GraAttenLayer, MultiHeadedAttention, adj_mx_from_edges

def create_edges(seq_length=1, num_nodes=29):

    edges = [
        # Hand connectivity
        [0, 1], [1, 2], [2, 3], [3, 4], 
        [0, 5], [5, 6], [6, 7], [7, 8], 
        [0, 9], [9, 10], [10, 11], [11, 12],
        [0, 13], [13, 14], [14, 15], [15, 16], 
        [0, 17], [17, 18], [18, 19], [19, 20]]
    if num_nodes == 29:
        # Object connectivity
        edges.extend([
        [21, 22],[22, 24], [24, 23], [23, 21],
        [25, 26], [26, 28], [28, 27], [27, 25],
        [21, 25], [22, 26], [23, 27], [24, 28]])

    edges = torch.tensor(edges, dtype=torch.long)
    return edges

class GraphUnpool(nn.Module):

    def __init__(self, in_nodes, out_nodes):
        super(GraphUnpool, self).__init__()
        self.fc = nn.Linear(in_features=in_nodes, out_features=out_nodes)        

    def forward(self, X):
        X = X.transpose(1, 2)
        X = self.fc(X)
        X = X.transpose(1, 2)
        return X


class MeshGraFormer(nn.Module):
    def __init__(self, initial_adj, coords_dim=(2, 3), hid_dim=128, num_layers=3, n_head=4,  dropout=0.1, n_pts=21, adj_matrix_root='./GraFormer/adj_matrix', device='cuda:1'):
        super(MeshGraFormer, self).__init__()
        self.n_layers = num_layers
        self.initial_adj = initial_adj
        
        if n_pts == 778:
            initial_pts = 21
            obj=''
        else:
            initial_pts = 29
            obj='Object'
        points_levels = [initial_pts, round(n_pts / 16), n_pts // 4, n_pts]
        self.mask = [torch.tensor([[[True] * points_levels[i]]]).to(device) for i in range(3)]
        
        self.adj = [initial_adj.to(device)]
        self.adj.extend([torch.from_numpy(scipy.sparse.load_npz(f'{adj_matrix_root}/hand{obj}{points_levels[i]}.npz').toarray()).float().to(device) for i in range(1, 4)])
        
        # features_levels = [coords_dim[0], 256, 32]
        
        _gconv_input = ChebConv(in_c=coords_dim[0], out_c=hid_dim, K=2)
        _gconv_layers1 = []
        _gconv_layers2 = []
        _attention_layer = []
        _unpooling_layer = []

        dim_model = hid_dim
        c = copy.deepcopy
        
        
        for i in range(num_layers):
            
            attn = MultiHeadedAttention(n_head, dim_model)
            gcn = GraphNet(in_features=dim_model, out_features=dim_model, n_pts=points_levels[i])    
            
            _attention_layer.append(GraAttenLayer(dim_model, c(attn), c(gcn), dropout))
            _gconv_layers1.append(_ResChebGC(adj=self.adj[i], input_dim=dim_model, output_dim=dim_model, hid_dim=dim_model, p_dropout=0.1))
            _gconv_layers2.append(_ResChebGC(adj=self.adj[i], input_dim=dim_model, output_dim=dim_model, hid_dim=dim_model, p_dropout=0.1))
            _unpooling_layer.append(GraphUnpool(points_levels[i], points_levels[i+1]))

        self.gconv_input = _gconv_input
        self.gconv_layers1 = nn.ModuleList(_gconv_layers1)
        self.gconv_layers2 = nn.ModuleList(_gconv_layers2)
        self.atten_layers = nn.ModuleList(_attention_layer)
        self.unpooling_layer = nn.ModuleList(_unpooling_layer)

        self.gconv_output = ChebConv(in_c=dim_model, out_c=coords_dim[1], K=2)

    def forward(self, x):
        out = self.gconv_input(x, self.initial_adj)
        for i in range(self.n_layers):
            out = self.atten_layers[i](out, self.mask[i])
            out = self.gconv_layers1[i](out)
            out = self.gconv_layers2[i](out)
            out = self.unpooling_layer[i](out)
            
        out = self.gconv_output(out, self.adj[-1])
        
        return out

if __name__ == '__main__':
    features = 3 + 1024
    num_points = 29
    x = torch.zeros((1, num_points, features))
    edges = create_edges(1, num_points)
    initial_adj = adj_mx_from_edges(num_pts=num_points, edges=edges, sparse=False)
    mesh_graformer = MeshGraFormer(initial_adj, coords_dim=(features,3), n_pts=1778)
    output = mesh_graformer(x)
    print(output.shape)

