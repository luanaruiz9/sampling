# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:37:09 2023

@author: Luana Ruiz
"""

# TO DOS:
# Multiple realizations
# Consolidate things in classes/functions - more or less ok
# Make things faster

# NEXT STEPS:
# Validate if padding normalization makes sense in reconstruction - I think it's ok
# Link prediction on ER graph?

import os
import datetime
import pickle as pkl
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
import aux_functions

thisFilename = 'cora' # This is the general name of all related files

saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename) 

#\\\ Create .txt to store the values of the setting parameters for easier
# reference when running multiple experiments
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
# Append date and time of the run to the directory, to avoid several runs of
# overwritting each other.
saveDir = saveDir + '-' + today
# Create directory 
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
    
# Check devices
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

n_realizations = 1
K = 5

dataset = Planetoid(root='/tmp/Cora', name='Cora')
graph_og = dataset[0]
#graph_og = graph_og.subgraph(torch.arange(500)) # comment it out
pre_defined_kwargs = {'eigvecs': False}
graph = Data(x=graph_og.x, edge_index=graph_og.edge_index, 
             edge_weight=graph_og.edge_weight, y=graph_og.y,**pre_defined_kwargs)
graph = graph.to(device)

# Vector to store eigenvector reconstruction errors
rec_error_w = np.zeros((n_realizations,K))
rec_error_random = np.zeros((n_realizations,K))

# Vectors to store test results
results_no_eigs = np.zeros(n_realizations)
results_eigs = np.zeros(n_realizations)
results_pe = np.zeros(n_realizations)
results_w_samp_pe = np.zeros(n_realizations)
results_random_samp_pe = np.zeros(n_realizations)

for r in range(n_realizations):
    
    split = T.RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1,
    )
    train_data, val_data, test_data = split(graph)
    
    model = SignNetLinkPredNet(dataset.num_features, 32, 32).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    model = train_link_predictor(model, train_data, val_data, optimizer, criterion)
    
    test_auc = eval_link_predictor(model, test_data)
    results_no_eigs[r] = test_auc
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
    edge_index = torch.cat((edge_index,torch.flip(edge_index,dims=(1,0))),dim=1)
    edge_weight = torch.cat((edge_weight,edge_weight))
    adj_sparse = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))
    adj = adj_sparse.to_dense()
    
    # Computing normalized Laplacian
    L = aux_functions.compute_laplacian(adj_sparse,num_nodes)
    
    #eigvals, V = torch.lobpcg(L,k=K)
    eigvals, V = torch.linalg.eig(adj)
    idx = torch.argsort(torch.abs(eigvals))
    V = V[:,idx[0:K]].type(torch.float32)
    eigvals = eigvals[idx[0:K]].type(torch.float32)
    
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
    results_eigs[r] = test_auc
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
    results_pe[r] = test_auc
    print(f"Test: {test_auc:.3f}")
    
    print()
    
    ##############################################################################
    ############################# Sampling! ######################################
    ##############################################################################
    
    print('Sampling with spectral proxies...')
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
    
    # Computing normalized Laplacian
    L_ind = aux_functions.compute_laplacian(adj_sparse_ind,num_nodes_ind)
    
    lam = eigvals[-1]
    L_aux = L_ind.cpu()
    k = 5
    m2 = 50
    
    s_vec = greedy(f, lam, L_aux, k, m2, exponent=5)
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
            idx = np.sort(idx)
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
    
    # Computing normalized Laplacian
    L_new = aux_functions.compute_laplacian(adj_sparse_new,num_nodes_new)
    
    K = 5
    eigvals_new, V_new = torch.lobpcg(L_new,k=K)
    V_new = V_new.type(torch.float32)
    V_rec = torch.zeros(num_nodes, K, device=device)
    
    for i in range(V_new.shape[1]):
        v = V_new[:,i]
        x0 = np.random.multivariate_normal(np.zeros(num_nodes),np.eye(num_nodes)/np.sqrt(num_nodes))
        x0[sampled_idx] = v.cpu().numpy()*np.sqrt(m2*m3)/np.sqrt(num_nodes)
        v_padded_lb = -torch.ones(num_nodes, device=device)
        v_padded_lb[sampled_idx] = v*np.sqrt(m2*m3)/np.sqrt(num_nodes)
        v_padded_ub = torch.ones(num_nodes, device=device)
        v_padded_ub[sampled_idx] = v*np.sqrt(m2*m3)/np.sqrt(num_nodes)
        V_rec[:,i] = torch.from_numpy(reconstruct(f_rec, x0, v_padded_lb, v_padded_ub, L, k))
        rec_error_w[r,i] = torch.linalg.norm(V_rec[:,i]-V[:,i]/V[:,i])/torch.linalg.norm(V[:,i])
    
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
    results_w_samp_pe[r] = test_auc
    print(f"Test: {test_auc:.3f}")
    
    print()
    
    ##############################################################################
    ############################# Sampling! ######################################
    ##############################################################################
    
    print('Sampling at random...')
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
    
    # Computing normalized Laplacian
    L_ind = aux_functions.compute_laplacian(adj_sparse_ind,num_nodes_ind)
    
    lam = eigvals[-1]
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
            idx = np.sort(idx)
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
    
    # Computing normalized Laplacian
    L_new = aux_functions.compute_laplacian(adj_sparse_new,num_nodes_new)
    
    K = 5
    eigvals_new, V_new = torch.lobpcg(L_new,k=K)
    V_new = V_new.type(torch.float32)
    V_rec = torch.zeros(num_nodes, K, device=device)
    
    for i in range(V_new.shape[1]):
        v = V_new[:,i]
        x0 = np.random.multivariate_normal(np.zeros(num_nodes),np.eye(num_nodes)/np.sqrt(num_nodes))
        x0[sampled_idx] = v.cpu().numpy()*np.sqrt(m2*m3)/np.sqrt(num_nodes)
        v_padded_lb = -torch.ones(num_nodes, device=device)
        v_padded_lb[sampled_idx] = v*np.sqrt(m2*m3)/np.sqrt(num_nodes)
        v_padded_ub = torch.ones(num_nodes, device=device)
        v_padded_ub[sampled_idx] = v*np.sqrt(m2*m3)/np.sqrt(num_nodes)    
        V_rec[:,i] = torch.from_numpy(reconstruct(f_rec, x0, v_padded_lb, v_padded_ub, L, k))
        rec_error_random[r,i] = torch.linalg.norm(V_rec[:,i]-V[:,i]/V[:,i])/torch.linalg.norm(V[:,i])
    
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
    results_random_samp_pe[r] = test_auc
    print(f"Test: {test_auc:.3f}")
    
    print()
    
# Pickling

pkl.dump(results_no_eigs, open(os.path.join(saveDir,'results.p'), "wb"))
pkl.dump(results_eigs, open(os.path.join(saveDir,'results.p'), "wb"))
pkl.dump(results_pe, open(os.path.join(saveDir,'results.p'), "wb"))
pkl.dump(results_w_samp_pe , open(os.path.join(saveDir,'results.p'), "wb"))
pkl.dump(results_random_samp_pe, open(os.path.join(saveDir,'results.p'), "wb"))

pkl.dump(rec_error_w , open(os.path.join(saveDir,'rec_error.p'), "wb"))
pkl.dump(rec_error_random, open(os.path.join(saveDir,'rec_error.p'), "wb"))
