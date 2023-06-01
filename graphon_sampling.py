# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 07:42:16 2023

@author: Luana Ruiz
"""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, is_undirected

def generate_induced_graphon(og_graph, n_intervals):
    # Here, we are actually generating the graph associated with the induced graphon.
    # This explains the normalizations used here.
    n = og_graph.num_nodes
    
    n_nodes_per_int, n_nodes_last_int = np.divmod(n,n_intervals)
    edge_index = og_graph.edge_index
    
    if og_graph.edge_weight is not None:
        edge_weight = og_graph.edge_weight
    else:
        edge_weight = torch.ones(edge_index.shape[1],device=edge_index.device)
        
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
                                             j*n_nodes_per_int:(j+1)*n_nodes_per_int])/n_nodes_per_int
            elif i < n_intervals-1 and j == n_intervals-1:
                adj_ind_graphon[i,j] = torch.sum(adj[i*n_nodes_per_int:(i+1)*n_nodes_per_int,
                                             j*n_nodes_per_int:-1])/np.sqrt(n_nodes_per_int*n_nodes_last_int)
            elif j < n_intervals-1 and i == n_intervals-1:
                adj_ind_graphon[i,j] = torch.sum(adj[i*n_nodes_per_int:-1,
                                             j*n_nodes_per_int:(j+1)*n_nodes_per_int])/np.sqrt(n_nodes_per_int*n_nodes_last_int)
            else:
                adj_ind_graphon[i,j] = torch.sum(adj[i*n_nodes_per_int:-1,
                                             j*n_nodes_per_int:-1])/n_nodes_last_int
        # The signals below are not used, for now. Note that y is categorical.
        if i < n_intervals-1:
            x_ind_graphon[i] = torch.sum(x[i*n_nodes_per_int:(i+1)*n_nodes_per_int],axis=0)/np.sqrt(n_nodes_per_int)
            y_ind_graphon[i] = torch.sum(y[i*n_nodes_per_int:(i+1)*n_nodes_per_int])/n_nodes_per_int > 0.5
        else:
            x_ind_graphon[i] = torch.sum(x[i*n_nodes_per_int:-1],axis=0)/np.sqrt(n_nodes_last_int)
            y_ind_graphon[i] = torch.sum(y[i*n_nodes_per_int:-1])/n_nodes_last_int > 0.5
    edge_index_ind_graphon, edge_weight_ind_graphon = dense_to_sparse(adj_ind_graphon)
    assert is_undirected(edge_index_ind_graphon)

    data = Data(x=x_ind_graphon,edge_index=edge_index_ind_graphon,
                edge_weight=edge_weight_ind_graphon,y=y_ind_graphon)
    
    return data