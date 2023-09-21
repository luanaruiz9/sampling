# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:34:08 2023

@author: Luana Ruiz
"""

from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.nn import aggr
import torch
import scipy
import torch.nn as nn
import math

import torch.nn.functional as F

def LSIGF(weights, S, x):
    '''
    weights is a list of length k, with each element of shape d_in x d_out
    S is N x N, sparse matrix
    x is N x K x d, d-dimensional feature (i.e., unique node ID in the featureless setting)
    '''    
    # Number of filter taps
    K = len(weights)

    # Create list to store graph diffused signals
    zs = [x]
    
    # Loop over the number of filter taps / different degree of S
    for k in range(1, K):        
        # diffusion step, S^k*x
        x = torch.matmul(torch.permute(x,(1,2,0)),S) #torch.matmul(x, S) -- slow
        # append the S^k*x in the list z
        x = torch.permute(x,(2,0,1))
        zs.append(x)
    
    # sum up
    out = [z @ weight for z, weight in zip(zs, weights)]
    out = torch.stack(out)
    y = torch.sum(out, axis=0)
    return y

class GraphFilter(torch.nn.Module):

    def __init__(self, Fin, Fout, K, normalize=True):
        super(GraphFilter, self).__init__()
        self.Fin = Fin 
        self.Fout = Fout
        self.K = K
        self.normalize = normalize
        self.weight = nn.ParameterList([nn.Parameter(torch.randn(self.Fin,self.Fout)) for k in range(self.K)])
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Fin * self.K)
        for elem in self.weight:
          elem.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index, edge_weight):
        N = x.shape[0]
        E = edge_index.shape[1]
        if edge_weight is None:
            edge_weight = torch.ones(E)
            edge_weight = edge_weight.to(x.device)
        S = torch.sparse_coo_tensor(edge_index, edge_weight, (N,N))

        if self.normalize:
            edge_weight_np = edge_weight.cpu().numpy()
            edge_index_np = edge_index.cpu().numpy()
            aux = scipy.sparse.coo_matrix((edge_weight_np, (edge_index_np[0],edge_index_np[1])), shape=(N,N))
            u, s, vh = scipy.sparse.linalg.svds(aux, k=1)
            S = S/torch.tensor(s[0]).to(S.device)

        return LSIGF(self.weight,S.to_dense(),x)


class GNN(torch.nn.Module):

    def __init__(self, GNNtype, Flist, MLPlist, softmax=False, Klist=None, 
                 aggregate=False, num_graph_classes=None):

        super(GNN, self).__init__()
        self.type = GNNtype
        self.Flist = Flist
        self.L = len(Flist)
        self.MLPlist = MLPlist 
        self.Lmlp = len(MLPlist)
        self.softmax = softmax
        self.Klist = Klist
        self.aggregate = aggregate
        if self.aggregate:
            self.aggregator = aggr.MeanAggregation()
            self.softmax = True
            if len(MLPlist) > 0:
                in_feats = MLPlist[-1]
            else:
                in_feats = Flist[-1]
            self.lin = nn.Linear(in_feats,num_graph_classes)
        self.layers = nn.ModuleList()
        self.MLPlayers = nn.ModuleList()

        for i in range(self.L-1):
            if self.type == 'sage':
                self.layers.append(SAGEConv(Flist[i],Flist[i+1]))
            elif self.type == 'gcn':
                self.layers.append(GCNConv(Flist[i],Flist[i+1]))
            elif self.type =='gnn':
                self.layers.append(GraphFilter(Flist[i],Flist[i+1],Klist[i]))

        for i in range(self.Lmlp-1):
            self.MLPlayers.append(nn.Linear(MLPlist[i],MLPlist[i+1]))
            
    def forward(self, data):

        y, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_weight, data.batch
        for i, layer in enumerate(self.layers):
            if self.type == 'gnn':
                y = layer(y, edge_index=edge_index, edge_weight=edge_weight)
            else:
                y = layer(y, edge_index=edge_index)
            y = F.relu(y)

        for i, layer in enumerate(self.MLPlayers):
            y = layer(y)
        
        if self.aggregate:
            y = self.aggregator(y, batch)
            y = F.dropout(y, p=0.5, training=self.training)
            y = self.lin(y)
            
        if self.softmax:
            y = F.log_softmax(y, dim=-1)
        return y

class LinkPredNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(
            dim=-1
        )  # product of a pair of nodes on each edge

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()
    

class SignNetLinkPredNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, sign=False,
                 in_channels_sign=None, hidden_channels_sign=None, out_channels_sign=None):
        super().__init__()
        if sign:
            self.conv1 = GraphFilter(in_channels_sign, in_channels_sign, 5)
            self.conv2 = GraphFilter(in_channels_sign, out_channels_sign, 1)
        self.inner_model = LinkPredNet(in_channels, hidden_channels, out_channels)
        
    def encode(self, x, edge_index, eigvecs=None):
        if eigvecs is not False:
            eigvecs = eigvecs.unsqueeze(-1)
            pe = eigvecs*self.conv1(torch.abs(eigvecs), edge_index, None).relu()
            pe = self.conv2(pe,edge_index, None)
            pe = pe.reshape(pe.shape[0],-1)
            x = torch.cat((x, pe),dim=1)
        return self.inner_model.encode(x, edge_index)
    def decode(self, z, edge_label_index):
        return self.inner_model.decode(z, edge_label_index)
    
