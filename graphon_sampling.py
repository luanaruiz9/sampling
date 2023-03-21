# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 07:42:16 2023

@author: Luana Ruiz
"""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

def generate_induced_graphon(og_graph, n_intervals):
    n = og_graph.num_nodes
    
    n_nodes_per_int, n_nodes_last_int = np.divmod(n,n_intervals)
    edge_index = og_graph.edge_label_index
    
    edge_weight = og_graph.edge_label
    if edge_weight is None or edge_weight.shape[0] != edge_index.shape[1]:
        edge_weight = torch.ones(edge_index.shape[1],device=edge_index.device)
    edge_index = torch.cat((edge_index,torch.flip(edge_index,dims=(1,0))),dim=1)
    edge_weight = torch.cat((edge_weight,edge_weight))
        
    adj_sparse = torch.sparse_coo_tensor(edge_index, edge_weight, (n, n))
    adj = adj_sparse.to_dense()
    adj_ind_graphon = torch.zeros(n_intervals,n_intervals,device=edge_index.device)
    x = og_graph.x
    x_ind_graphon = torch.zeros(n_intervals,og_graph.x.shape[1],device=edge_index.device)
    y = og_graph.y
    y_ind_graphon = torch.zeros(n_intervals,device=edge_index.device)
    
    for i in range(n_intervals):
        for j in range(n_intervals):
            if i < n_intervals-1 and j < n_intervals-1:
                adj_ind_graphon[i,j] = torch.sum(adj[i*n_nodes_per_int:(i+1)*n_nodes_per_int,
                                             j*n_nodes_per_int:(j+1)*n_nodes_per_int])/(n_nodes_per_int*n_nodes_per_int)
            elif i < n_intervals-1 and j == n_intervals-1:
                adj_ind_graphon[i,j] = torch.sum(adj[i*n_nodes_per_int:(i+1)*n_nodes_per_int,
                                             j*n_nodes_per_int:-1])/(n_nodes_last_int*n_nodes_last_int)
            else:
                adj_ind_graphon[i,j] = torch.sum(adj[i*n_nodes_per_int:-1,
                                             j*n_nodes_per_int:(j+1)*n_nodes_per_int])/(n_nodes_last_int*n_nodes_last_int)
        if i < n_intervals-1:
            x_ind_graphon[i] = torch.sum(x[i*n_nodes_per_int:(i+1)*n_nodes_per_int],axis=0)/n_nodes_per_int
            y_ind_graphon[i] = torch.sum(y[i*n_nodes_per_int:(i+1)*n_nodes_per_int])/n_nodes_per_int
        else:
            x_ind_graphon[i] = torch.sum(x[i*n_nodes_per_int:-1],axis=0)/n_nodes_per_int
            y_ind_graphon[i] = torch.sum(y[i*n_nodes_per_int:-1])/n_nodes_last_int
    edge_index_ind_graphon, edge_weight_ind_graphon = dense_to_sparse(adj_ind_graphon)

    data = Data(x=x_ind_graphon,edge_index=edge_index_ind_graphon,
                edge_weight=edge_weight_ind_graphon,y=y_ind_graphon)
    
    return data