# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:18:48 2023

@author: Luana Ruiz
"""

import torch


def compute_adj_from_data(data):
    num_nodes = data.x.shape[0]
    device = data.x.device
    edge_index = data.edge_index
    if data.edge_weight is not None:
        edge_weight = data.edge_weight
    else:
        edge_weight = torch.ones(edge_index.shape[1]).to(device)
    adj_sparse = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))
    adj = adj_sparse.to_dense()
    return adj_sparse, adj

def compute_degree(adj_sparse, num_nodes):
    device = adj_sparse.device
    degree = torch.matmul(adj_sparse,torch.ones(num_nodes).to(device))
    edge_index_deg = torch.cat((torch.arange(num_nodes).unsqueeze(0),torch.arange(num_nodes).unsqueeze(0)),dim=0)
    degree_mx = torch.sparse_coo_tensor(edge_index_deg.to(device), degree, (num_nodes, num_nodes))
    return degree_mx
    
def compute_inv_degree(adj_sparse, num_nodes):
    device = adj_sparse.device
    degree = torch.pow(torch.matmul(adj_sparse,torch.ones(num_nodes).to(device)),-0.5)
    #degree = torch.nan_to_num(degree) 
    edge_index_deg = torch.cat((torch.arange(num_nodes).unsqueeze(0),torch.arange(num_nodes).unsqueeze(0)),dim=0)
    degree_mx = torch.sparse_coo_tensor(edge_index_deg.to(device), degree, (num_nodes, num_nodes))
    return degree_mx

def compute_laplacian(adj_sparse, num_nodes):
    degree_mx = compute_inv_degree(adj_sparse,num_nodes)
    L = torch.eye(num_nodes,device=adj_sparse.device).to_sparse_coo()-torch.matmul(degree_mx,torch.matmul(adj_sparse,degree_mx))
    return L

