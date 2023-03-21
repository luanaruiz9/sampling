# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:37:09 2023

@author: Luana Ruiz
"""

# FIX: Eigenvectors can only come from training set!!!!!!
# FIX: Test of sampling (need reconstruction)
# Link prediction on ER graph?


# TO DO: recall AUC - ok
# TO DO: train a second model on data + eigenvectors - ok
# TO DO: train a third model on data + PEs (Derek's code? or GCN/GIN + SignNet)
# TO DO: Implement greedy sampling
# TO DO: Figure out dual frame or approximately dual frame - variational splines? 
# other types of splines?
# TO DO: Implement reconstruction and test

import numpy as np

import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.data import Data

from architecture import LinkPredNet, SignNetLinkPredNet
from train_eval import train_link_predictor, eval_link_predictor
from greedy import greedy, f
from graphon_sampling import generate_induced_graphon

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

dataset = Planetoid(root='/tmp/Cora', name='Cora')
graph_og = dataset[0]
#graph_og = graph_og.subgraph(torch.arange(500)) # comment it out
pre_defined_kwargs = {'eigvecs': False}
graph = Data(x=graph_og.x, edge_index=graph_og.edge_index, 
             edge_weight=graph_og.edge_weight, y=graph_og.y,**pre_defined_kwargs)
graph = graph.to(device)
split = T.RandomLinkSplit(
    num_val=0.05,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=False,
    neg_sampling_ratio=1.0,
)
train_data, val_data, test_data = split(graph)

model = SignNetLinkPredNet(dataset.num_features, 32, 32).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
model = train_link_predictor(model, train_data, val_data, optimizer, criterion)

test_auc = eval_link_predictor(model, test_data)
print(f"Test: {test_auc:.3f}")

print()

##############################################################################
######################## Adding eigenvectors #################################
##############################################################################

print('Adding eigenvectors...')
print()

num_nodes = graph.x.shape[0]
edge_index = graph.edge_index
if graph.edge_weight is not None:
    edge_weight = graph.edge_weight
else:
    edge_weight = torch.ones(edge_index.shape[1]).to(device)
adj_sparse = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))
adj = adj_sparse.to_dense()

# Computing degree
degree = torch.pow(torch.matmul(adj_sparse,torch.ones(num_nodes).to(device)),-0.5)
edge_index_deg = torch.cat((torch.arange(num_nodes).unsqueeze(0),torch.arange(num_nodes).unsqueeze(0)),dim=0)
degree_mx = torch.sparse_coo_tensor(edge_index_deg.to(device), degree, (num_nodes, num_nodes))

# Computing normalized Laplacian

L = torch.matmul(degree_mx,torch.matmul(adj_sparse,degree_mx))

K = 5
eigvals, V = torch.lobpcg(L,k=K)
#L, V = torch.linalg.eig(adj)
#idx = torch.argsort(-torch.abs(L))
#V = V[:,idx[0:K]]
x_new = torch.cat((graph.x,V), dim=1)
pre_defined_kwargs = {'eigvecs': False}
graph_new = Data(x=x_new,edge_index=edge_index,edge_weight=graph.edge_index,
                 y=graph.y,**pre_defined_kwargs)
graph_new = graph_new.to(device)

split = T.RandomLinkSplit(
    num_val=0.05,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=False,
    neg_sampling_ratio=1.0,
)
train_data, val_data, test_data = split(graph_new)

model = SignNetLinkPredNet(dataset.num_features+K, 32, 32).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
model = train_link_predictor(model, train_data, val_data, optimizer, criterion)

test_auc = eval_link_predictor(model, test_data)
print(f"Test: {test_auc:.3f}")

print()

##############################################################################
############################# Adding PEs #####################################
##############################################################################

print('Adding PE...')
print()

x_new = graph_og.x
pre_defined_kwargs = {'eigvecs': V}
graph_new = Data(x=x_new,edge_index=edge_index,edge_weight=graph.edge_index,
                 y=graph.y,**pre_defined_kwargs)
graph_new = graph_new.to(device)

split = T.RandomLinkSplit(
    num_val=0.05,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=False,
    neg_sampling_ratio=1.0,
)
train_data, val_data, test_data = split(graph_new)
model = SignNetLinkPredNet(dataset.num_features+16*K, 32, 32, True, 1, 16, 16).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
model = train_link_predictor(model, train_data, val_data, optimizer, criterion)
test_auc = eval_link_predictor(model, test_data)
print(f"Test: {test_auc:.3f}")

print()

##############################################################################
############################# Sampling! ######################################
##############################################################################

print('Sampling...')
print()

# Finding sampling set

m = 677
graph_ind, _, _ = generate_induced_graphon(graph_og,m)
num_nodes_ind = graph_ind.x.shape[0]
edge_index_ind = graph_ind.edge_index
edge_weight_ind = graph_ind.edge_weight
adj_sparse_ind = torch.sparse_coo_tensor(edge_index_ind, edge_weight_ind,
                                         (num_nodes_ind, num_nodes_ind))
adj_ind = adj_sparse_ind.to_dense()
adj_ind = adj_ind.to(device)

# Computing degree
degree = torch.pow(torch.matmul(adj_sparse_ind,torch.ones(num_nodes_ind).to(device)),-0.5)
edge_index_deg = torch.cat((torch.arange(num_nodes_ind).unsqueeze(0),
                            torch.arange(num_nodes_ind).unsqueeze(0)),dim=0)
degree_mx = torch.sparse_coo_tensor(edge_index_deg.to(device), degree, 
                                    (num_nodes_ind, num_nodes_ind))

# Computing normalized Laplacian

L_ind = torch.matmul(degree_mx,torch.matmul(adj_sparse_ind,degree_mx))

lam = 0.5 
L_aux = L_ind.cpu()
k = 5
m2 = 50

s_vec = greedy(f, lam, L_aux, k, m2)
s_vec = torch.tensor(s_vec)

graph_new = graph_og.subgraph(torch.argwhere(s_vec))
graph_new = graph_new.to(device)

num_nodes_new = graph_new.x.shape[0]
edge_index_new = graph_new.edge_index
if graph_new.edge_weight is not None:
    edge_weight_new = graph_new.edge_weight
else:
    edge_weight_new = torch.ones(edge_index_new.shape[1]).to(device)
adj_sparse_new = torch.sparse_coo_tensor(edge_index_new, edge_weight_new, 
                                         (num_nodes_new, num_nodes_new))
adj_new = adj_sparse_new.to_dense()

# Computing degree
degree = torch.pow(torch.matmul(adj_sparse_new,torch.ones(num_nodes_new).to(device)),-0.5)
edge_index_deg = torch.cat((torch.arange(num_nodes_new).unsqueeze(0),
                            torch.arange(num_nodes_new).unsqueeze(0)),dim=0)
degree_mx_new = torch.sparse_coo_tensor(edge_index_deg.to(device), degree, (num_nodes_new, num_nodes_new))

# Computing normalized Laplacian

L_new = torch.matmul(degree_mx_new,torch.matmul(adj_sparse_new,degree_mx_new))

K = 5
eigvals_new, V_new = torch.lobpcg(L_new,k=K)

"""
x_new = graph_new.x
pre_defined_kwargs = {'eigvecs': V_new}
graph_new = Data(x=x_new,edge_index=edge_index_new,edge_weight=graph_new.edge_index,
                 y=graph_new.y,**pre_defined_kwargs)

split = T.RandomLinkSplit(
    num_val=0.05,
    num_test=0.1,
    is_undirected=True,
    add_negative_train_samples=False,
    neg_sampling_ratio=1.0,
)
train_data, val_data, test_data = split(graph_new) # test data is data from large, original graph
model = SignNetLinkPredNet(dataset.num_features+16*K, 32, 32, True, 1, 16, 16).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
model = train_link_predictor(model, train_data, val_data, optimizer, criterion)
test_auc = eval_link_predictor(model, test_data)
print(f"Test: {test_auc:.3f}")

print()
"""