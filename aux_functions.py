# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:18:48 2023

@author: Luana Ruiz
"""
import torch

def compute_degree(adj_sparse,num_nodes):
    device = adj_sparse.device
    degree = torch.pow(torch.matmul(adj_sparse,torch.ones(num_nodes).to(device)),-0.5)
    edge_index_deg = torch.cat((torch.arange(num_nodes).unsqueeze(0),torch.arange(num_nodes).unsqueeze(0)),dim=0)
    degree_mx = torch.sparse_coo_tensor(edge_index_deg.to(device), degree, (num_nodes, num_nodes))
    return degree_mx
    
    
def compute_laplacian(adj_sparse,num_nodes):
    degree_mx = compute_degree(adj_sparse,num_nodes)
    L = torch.matmul(degree_mx,torch.matmul(adj_sparse,degree_mx))
    return L
