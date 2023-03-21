# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:37:09 2023

@author: Luana Ruiz
"""

# Check if there are any graphon normalizations I'm missing - probably in reconstruction
# Link prediction on ER graph?
# What is link sampling doing that I only have half the edges?


# TO DO: recall AUC - ok
# TO DO: train a second model on data + eigenvectors - ok
# TO DO: train a third model on data + PEs (Derek's code? or GCN/GIN + SignNet)
# TO DO: Implement greedy sampling - ok
# TO DO: Figure out dual frame or approximately dual frame - variational splines?  
# other types of splines? - ok
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
from reconstruction import f_rec, reconstruct

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

num_nodes = train_data.x.shape[0]
edge_index = train_data.edge_label_index
if train_data.edge_label is not None:
    edge_weight = train_data.edge_label
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
#eigvals, V = torch.lobpcg(L,k=K)
eigvals, V = torch.linalg.eig(adj)
idx = torch.argsort(-torch.abs(eigvals))
V = V[:,idx[0:K]].type(torch.float32)

pre_defined_kwargs = {'eigvecs': False}
train_data_new = Data(x=torch.cat((train_data.x,V), dim=1), edge_index=train_data.edge_index,
                      edge_label=train_data.edge_label,
                      y=train_data.y,edge_label_index=train_data.edge_label_index,
                      **pre_defined_kwargs)
val_data_new = Data(x=torch.cat((val_data.x,V), dim=1), edge_index=val_data.edge_index,
                      edge_label=val_data.edge_label,
                      y=train_data.y,edge_label_index=val_data.edge_label_index,
                      **pre_defined_kwargs)
test_data_new = Data(x=torch.cat((test_data.x,V), dim=1), edge_index=test_data.edge_index,
                      edge_label=test_data.edge_label,
                      y=test_data.y,edge_label_index=test_data.edge_label_index,
                      **pre_defined_kwargs)

model = SignNetLinkPredNet(dataset.num_features+K, 32, 32).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
model = train_link_predictor(model, train_data_new, val_data_new, optimizer, criterion)

test_auc = eval_link_predictor(model, test_data_new)
print(f"Test: {test_auc:.3f}")

print()

##############################################################################
############################# Adding PEs #####################################
##############################################################################

print('Adding PE...')
print()

pre_defined_kwargs = {'eigvecs': V}
train_data_new = Data(x=train_data.x, edge_index=train_data.edge_index,
                      edge_label=train_data.edge_label,
                      y=train_data.y,edge_label_index=train_data.edge_label_index,
                      **pre_defined_kwargs)
val_data_new = Data(x=val_data.x, edge_index=val_data.edge_index,
                      edge_label=val_data.edge_label,
                      y=train_data.y,edge_label_index=val_data.edge_label_index,
                      **pre_defined_kwargs)
test_data_new = Data(x=test_data.x, edge_index=test_data.edge_index,
                      edge_label=test_data.edge_label,
                      y=test_data.y,edge_label_index=test_data.edge_label_index,
                      **pre_defined_kwargs)

model = SignNetLinkPredNet(dataset.num_features+16*K, 32, 32, True, 1, 16, 16).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
model = train_link_predictor(model, train_data_new, val_data_new, optimizer, criterion)
test_auc = eval_link_predictor(model, test_data_new)
print(f"Test: {test_auc:.3f}")

print()

##############################################################################
############################# Sampling! ######################################
##############################################################################

print('Sampling...')
print()

# Finding sampling set
m = 100
n_nodes_per_int, n_nodes_last_int = np.divmod(num_nodes,m)
graph_ind = generate_induced_graphon(train_data,m)
num_nodes_ind = graph_ind.x.shape[0]
edge_index_ind = graph_ind.edge_index
edge_weight_ind = graph_ind.edge_weight
adj_sparse_ind = torch.sparse_coo_tensor(edge_index_ind, edge_weight_ind,
                                         (num_nodes_ind, num_nodes_ind))
adj_sparse_ind = adj_sparse_ind.to(device)
adj_ind = adj_sparse_ind.to_dense()

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

s_vec = greedy(f, lam, L_aux, k, m2, exponent=0)
s_vec = torch.tensor(s_vec)

m3 = 4
sampled_idx = []
for i in range(m):
    if s_vec[i] == 1:
        if i < m-1:
            idx = np.random.choice(np.arange(i*n_nodes_per_int,(i+1)*n_nodes_per_int),m3)
        else:
            if m3 > n_nodes_last_int:
                m3 = n_nodes_last_int
            idx = np.random.choice(np.arange(i*n_nodes_per_int,
                                             i*n_nodes_per_int+n_nodes_last_int),m3)
        sampled_idx += list(idx)

graph_new = train_data.subgraph(torch.tensor(sampled_idx,device=device,dtype=torch.long))
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
V_new = V_new.type(torch.float32)
V_rec = torch.zeros(num_nodes, K, device=device)

for i in range(V_new.shape[1]):
    v = V_new[:,i]
    v_padded = torch.zeros(num_nodes, device=device)
    v_padded[sampled_idx] = v
    V_rec[:,i] = torch.from_numpy(reconstruct(f_rec, v_padded, v_padded, L, k))

pre_defined_kwargs = {'eigvecs': V_rec}

train_data_new = Data(x=train_data.x, edge_index=train_data.edge_index,
                      edge_label=train_data.edge_label,
                      y=train_data.y,edge_label_index=train_data.edge_label_index,
                      **pre_defined_kwargs)
val_data_new = Data(x=val_data.x, edge_index=val_data.edge_index,
                      edge_label=val_data.edge_label,
                      y=train_data.y,edge_label_index=val_data.edge_label_index,
                      **pre_defined_kwargs)
test_data_new = Data(x=test_data.x, edge_index=test_data.edge_index,
                      edge_label=test_data.edge_label,
                      y=test_data.y,edge_label_index=test_data.edge_label_index,
                      **pre_defined_kwargs)

model = SignNetLinkPredNet(dataset.num_features+16*K, 32, 32, True, 1, 16, 16).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()
model = train_link_predictor(model, train_data_new, val_data_new, optimizer, criterion)
test_auc = eval_link_predictor(model, test_data_new)
print(f"Test: {test_auc:.3f}")

print()
