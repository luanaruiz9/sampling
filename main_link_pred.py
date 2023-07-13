# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:37:09 2023

@author: Luana Ruiz
"""

# TO DOS:
# Consolidate things in classes/functions - more or less ok
# Make things faster

# NEXT STEPS:
# Link prediction on synthetic graphs
# Transferability experiments

import sys
import os
import datetime
import pickle as pkl
import numpy as np

import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Twitch, StochasticBlockModelDataset
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes, to_networkx, add_self_loops, to_undirected
import networkx as nx

from architecture import  SignNetLinkPredNet
from train_eval import train_link_predictor, eval_link_predictor
from sampling import generate_induced_graphon, greedy, f, sample_clustering
#from reconstruction import f_rec, reconstruct
import aux_functions

data_name = sys.argv[1]
lr = float(sys.argv[2])
n_epochs = int(sys.argv[3])
ratio_train = 0.8
ratio_test = 0.1
ratio_val = 1-ratio_train-ratio_test
n_realizations = int(sys.argv[4]) #10
m = int(sys.argv[5]) #50 # Number of candidate intervals
m2 = int(sys.argv[6]) #25 # Number of sampled intervals
m3 = int(sys.argv[7]) #3 #8 # How many nodes (points) to sample per sampled interval
updated_sz = m2*m3
nb_cuts = int(sys.argv[8])
fnn = int(sys.argv[9])
fpe = int(sys.argv[10])

F_nn = [fnn, fnn]
F_pe = [fpe, fpe]

thisFilename = data_name + '_' # This is the general name of all related files

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
    
do_no_pe = True
do_eig = True
do_learn_pe = True
do_w_sampl = True
do_random_sampl = True

remove_isolated = False

if 'cora' in data_name:
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
elif 'citeseer' in data_name:
    dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
elif 'pubmed' in data_name:
    dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
elif 'chameleon' in data_name:
    dataset = WikipediaNetwork(root='/tmp/Chameleon', name='chameleon')
elif 'twitch-pt' in data_name:
    dataset = Twitch(root='/tmp/PT', name='PT')
elif 'twitch-ru' in data_name:
    dataset = Twitch(root='/tmp/RU', name='RU')
elif 'sbm-d' in data_name:
    n = 20000
    c = 50
    b_sz = int(n/c)*torch.ones(c,dtype=torch.long)
    q = 0.3
    p = 0.7
    p_m = q*torch.ones(c,c) + (p-q)*torch.eye(c)
    dataset = StochasticBlockModelDataset(root='/tmp/SBM-d', block_sizes=b_sz, edge_probs=p_m)
elif 'sbm-s' in data_name:
    n = 20000
    c = 50
    b_sz = int(n/c)*torch.ones(c,dtype=torch.long)
    q = 0.3
    p = 0.7
    p_m = (q*torch.ones(c,c) + (p-q)*torch.eye(c))*(np.log(n)/n)
    dataset = StochasticBlockModelDataset(root='/tmp/SBM-s', block_sizes=b_sz, edge_probs=p_m)
    
graph_og = dataset[0]
#graph_og = graph_og.subgraph(torch.arange(500)) # comment it out

# Sorting nodes by degree
adj_sparse, adj = aux_functions.compute_adj_from_data(graph_og)
num_nodes = adj.shape[0]
D = aux_functions.compute_degree(adj_sparse, num_nodes)
deg = torch.diagonal(D.to_dense()).squeeze()
idx = torch.argsort(deg)
edge_index = graph_og.edge_index
new_edge_index = torch.zeros(edge_index.shape,dtype=torch.long,device=device)
for i in range(2):
    for j in range(edge_index.shape[1]):
        new_edge_index[i,j] = torch.argwhere(edge_index[i,j]==idx)
graph_og = Data(x=graph_og.x[idx],edge_index=new_edge_index,y=graph_og.y[idx])

pre_defined_kwargs = {'eigvecs': False}
graph = Data(x=torch.ones(graph_og.x.shape[0],1), edge_index=graph_og.edge_index, 
             edge_weight=graph_og.edge_weight, y=graph_og.y,**pre_defined_kwargs)
num_feats = graph.x.shape[1]
graph = graph.to(device)

# Vectors to store test results
results_no_eigs = np.zeros(n_realizations)
results_eigs = np.zeros(n_realizations)
results_pe = np.zeros(n_realizations)
results_w_samp_eigs = np.zeros(n_realizations)
results_w_samp_pe = np.zeros(n_realizations)
results_random_samp_eigs = np.zeros(n_realizations)
results_random_samp_pe = np.zeros(n_realizations)
n_iters_per_rlz = np.zeros(n_realizations)
len_sampled_idx = np.zeros(n_realizations)
len_sampled_idx2 = np.zeros(n_realizations)

for r in range(n_realizations):
    K = 50
    print('Realization ' + str(r))
    print()
    
    split = T.RandomLinkSplit(
        num_val=ratio_val,
        num_test=ratio_test,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1,
    )

    train_data, val_data, test_data = split(graph)
    
    if do_no_pe:
    
        model = SignNetLinkPredNet(num_feats, F_nn[0], F_nn[1]).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        model, _, _ = train_link_predictor(model, train_data, val_data, optimizer, 
                                     criterion, n_epochs=n_epochs)
        
        test_auc = eval_link_predictor(model, test_data)
        results_no_eigs[r] = test_auc
        print(f"Test: {test_auc:.3f}")
        
        print()
    
    ##############################################################################
    ######################## Adding eigenvectors #################################
    ##############################################################################

    # V for train data
    adj_sparse, adj = aux_functions.compute_adj_from_data(train_data)
    num_nodes = adj.shape[0]
    
    # Computing normalized Laplacian
    L = aux_functions.compute_laplacian(adj_sparse, num_nodes)
    eigvals, V = torch.lobpcg(L, k=K, largest=False)
    #eigvals, V = torch.linalg.eig(L.to_dense())
    eigvals = eigvals.float()
    V = V.float()
    idx = torch.argsort(eigvals)
    eigvals = eigvals[idx[0:K]]
    V = V[:,idx[0:K]]
    
    # V for test data
    adj_sparse_test, adj_test = aux_functions.compute_adj_from_data(test_data)
    
    # Computing normalized Laplacian
    L_test = aux_functions.compute_laplacian(adj_sparse_test, num_nodes)
    eigvals_test, V_test = torch.lobpcg(L_test, k=K, largest=False)
    #eigvals_test, V_test = torch.linalg.eig(L_test.to_dense())
    eigvals_test = eigvals_test.float()
    V_test = V_test.float()
    idx = torch.argsort(eigvals_test)
    eigvals_test = eigvals_test[idx[0:K]]
    V_test = V_test[:,idx[0:K]]
    
    if do_eig:
    
        print('Adding eigenvectors...')
        print()
        
        pre_defined_kwargs = {'eigvecs': False}
        
        train_data_new = Data(x=train_data.x, edge_index=train_data.edge_index,
                              edge_label=train_data.edge_label,
                              y=train_data.y,edge_label_index=train_data.edge_label_index,
                              **pre_defined_kwargs)
        val_data_new = Data(x=torch.cat((val_data.x,V), dim=1), edge_index=val_data.edge_index,
                              edge_label=val_data.edge_label,
                              y=train_data.y,edge_label_index=val_data.edge_label_index,
                              **pre_defined_kwargs)
        test_data_new = Data(x=torch.cat((test_data.x,V_test), dim=1), edge_index=test_data.edge_index,
                              edge_label=test_data.edge_label,
                              y=test_data.y,edge_label_index=test_data.edge_label_index,
                              **pre_defined_kwargs)
        
        model = SignNetLinkPredNet(num_feats+K, F_nn[0], F_nn[1]).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        model, _, _ = train_link_predictor(model, train_data_new, val_data_new, optimizer, 
                                     criterion, n_epochs=n_epochs, K=K)
        
        test_auc = eval_link_predictor(model, test_data_new)
        results_eigs[r] = test_auc
        print(f"Test: {test_auc:.3f}")
        
        print()
    
    ##############################################################################
    ############################# Adding PEs #####################################
    ##############################################################################
    
    if do_learn_pe:
    
        print('Adding PE...')
        print()
        
        pre_defined_kwargs = {'eigvecs': V}
        pre_defined_kwargs_test = {'eigvecs': V_test}
        
        train_data_new = Data(x=train_data.x, edge_index=train_data.edge_index,
                              edge_label=train_data.edge_label,
                              y=train_data.y,edge_label_index=train_data.edge_label_index)
        val_data_new = Data(x=val_data.x, edge_index=val_data.edge_index,
                              edge_label=val_data.edge_label,
                              y=train_data.y,edge_label_index=val_data.edge_label_index,
                              **pre_defined_kwargs)
        test_data_new = Data(x=test_data.x, edge_index=test_data.edge_index,
                              edge_label=test_data.edge_label,
                              y=test_data.y,edge_label_index=test_data.edge_label_index,
                              **pre_defined_kwargs_test)
        
        model = SignNetLinkPredNet(num_feats+F_pe[-1]*K, F_nn[0], F_nn[1], True, 1, F_pe[0], F_pe[1]).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        model, _, _ = train_link_predictor(model, train_data_new, val_data_new, optimizer, 
                                     criterion, n_epochs=n_epochs, K=K, pe=True)
        test_auc = eval_link_predictor(model, test_data_new)
        results_pe[r] = test_auc
        print(f"Test: {test_auc:.3f}")
        
        print()
    
    ##############################################################################
    ############################# Sampling! ######################################
    ##############################################################################
    
    if do_w_sampl:
    
        print('Sampling with spectral proxies...')
        print()
        
        idx = torch.argsort(eigvals_test.float())
        V_test = V_test[:,idx[0:K]].type(torch.float32)
        eigvals_test = eigvals_test[idx[0:K]].type(torch.float32)
        
        # Finding sampling set
        n_nodes_per_int, n_nodes_last_int = np.divmod(num_nodes, m)
        graph_ind = generate_induced_graphon(train_data, m)
        num_nodes_ind = graph_ind.x.shape[0]
        assert num_nodes_ind == m
        adj_sparse_ind, adj_ind = aux_functions.compute_adj_from_data(graph_ind)
        
        # Computing normalized Laplacian
        L_ind = aux_functions.compute_laplacian(adj_sparse_ind,num_nodes_ind)
        
        lam = eigvals[-1]
        L_aux = L_ind.cpu()
        k = 5
        
        s_vec, n_iters = greedy(f, lam, L_aux, k, m2, exponent=100000000)
        n_iters_per_rlz[r] = n_iters
        s_vec = torch.tensor(s_vec)
        
        sampled_idx = []
        for i in range(m):
            if s_vec[i] == 1:
                if i < m-1:
                    cur_adj = adj[i*n_nodes_per_int:(i+1)*n_nodes_per_int,:]
                    cur_adj = cur_adj[:,i*n_nodes_per_int:(i+1)*n_nodes_per_int]
                    idx = sample_clustering(cur_adj, m3, nb_cuts=nb_cuts)#np.random.choice(np.arange(i*n_nodes_per_int,(i+1)*n_nodes_per_int), m3, replace=False)
                else:
                    if m3 > n_nodes_last_int:
                        #m3 = n_nodes_last_int
                        cur_adj = adj[i*n_nodes_per_int:i*n_nodes_per_int+n_nodes_last_int,:]
                        cur_adj = cur_adj[:,i*n_nodes_per_int:i*n_nodes_per_int+n_nodes_last_int]
                    else:
                        cur_adj = adj[i*n_nodes_per_int:i*n_nodes_per_int+n_nodes_last_int,:]
                        cur_adj = cur_adj[:,i*n_nodes_per_int:i*n_nodes_per_int+m3]
                    idx = sample_clustering(cur_adj, n_nodes_last_int, nb_cuts=nb_cuts)#np.random.choice(np.arange(i*n_nodes_per_int,
                                                     #i*n_nodes_per_int+n_nodes_last_int), m3, replace=False)
                idx = np.sort(idx)
                for j in range(idx.shape[0]):
                    idx[j] += i*n_nodes_per_int
                sampled_idx += list(idx)
        sampled_idx = list(set(sampled_idx))   
        updated_sz = len(sampled_idx)
        
        # V for train data
        graph_new = train_data.subgraph(torch.tensor(sampled_idx, device=device, dtype=torch.long))

        # Removing isolated nodes
        sampled_idx_og = sampled_idx
        if remove_isolated:
            edge_index_new = graph_new.edge_index.clone()
            edge_index_new, _, mask = remove_isolated_nodes(edge_index_new, num_nodes = len(sampled_idx_og))
            mask = mask.cpu().tolist()
            sampled_idx = list(np.array(sampled_idx_og)[mask])
            graph_new = graph_new.subgraph(torch.tensor(mask, device=device))
        if K > len(sampled_idx):
            K = len(sampled_idx)
        len_sampled_idx[r] = len(sampled_idx)

        graph_new = graph_new.to(device)
        num_nodes_new = graph_new.x.shape[0]
        adj_sparse_new, adj_new = aux_functions.compute_adj_from_data(graph_new)
        
        # Computing normalized Laplacian
        L_new = aux_functions.compute_laplacian(adj_sparse_new, num_nodes_new)
        
        #eigvals_new, V_new = torch.lobpcg(L_new, k=K, largest=False)
        eigvals_new, V_new = torch.linalg.eig(L_new.to_dense())
        eigvals_new = eigvals_new.float()
        V_new = V_new.float()
        idx = torch.argsort(eigvals_new)
        eigvals_new = eigvals_new[idx[0:K]]
        V_new = V_new[:,idx[0:K]]
        V_new = V_new.type(torch.float32)
        V_rec = torch.zeros(num_nodes, K, device=device)
        
        for i in range(V_new.shape[1]):
            v = V_new[:,i]
            V_rec[sampled_idx,i] = v
        
        # V for test data
        graph_new = test_data.subgraph(torch.tensor(sampled_idx_og, device=device, dtype=torch.long))
        
        # Removing isolated nodes
        if remove_isolated:
            edge_index_new = graph_new.edge_index.clone()
            edge_index_new, _, mask = remove_isolated_nodes(edge_index_new, num_nodes = len(sampled_idx_og))
            mask = mask.cpu().tolist()
            sampled_idx = list(np.array(sampled_idx_og)[mask])
            graph_new = graph_new.subgraph(torch.tensor(mask, device=device))
        if K > len(sampled_idx):
            K = len(sampled_idx)
        
        graph_new = graph_new.to(device)
        num_nodes_new = graph_new.x.shape[0]
        adj_sparse_new, adj_new = aux_functions.compute_adj_from_data(graph_new)
        
        # Computing normalized Laplacian
        L_new = aux_functions.compute_laplacian(adj_sparse_new, num_nodes_new)
        
        #eigvals_new, V_new = torch.lobpcg(L_new, k=K, largest=False)
        eigvals_new, V_new = torch.linalg.eig(L_new.to_dense())
        eigvals_new = eigvals_new.float()
        V_new = V_new.float()
        idx = torch.argsort(eigvals_new)
        eigvals_new = eigvals_new[idx[0:K]]
        V_new = V_new[:,idx[0:K]]
        V_new = V_new.type(torch.float32)
        V_rec_test = torch.zeros(num_nodes, K, device=device)
        
        for i in range(V_new.shape[1]):
            v = V_new[:,i]
            V_rec_test[sampled_idx,i] = v
            #rec_error_w[r,i] = torch.linalg.norm(V_rec_test[:,i]-V_test[:,i])/torch.linalg.norm(V_test[:,i])
        
        # Just adding eigenvectors
        
        print("Just adding eigenvectors...")
        print()
        
        pre_defined_kwargs = {'eigvecs': False}
        
        train_data_new = Data(x=train_data.x, edge_index=train_data.edge_index,
                              edge_label=train_data.edge_label,
                              y=train_data.y,edge_label_index=train_data.edge_label_index,
                              **pre_defined_kwargs)
        val_data_new = Data(x=torch.cat((val_data.x,V_rec), dim=1), edge_index=val_data.edge_index,
                              edge_label=val_data.edge_label,
                              y=train_data.y,edge_label_index=val_data.edge_label_index,
                              **pre_defined_kwargs)
        test_data_new = Data(x=torch.cat((test_data.x,V_rec_test), dim=1), edge_index=test_data.edge_index,
                              edge_label=test_data.edge_label,
                              y=test_data.y,edge_label_index=test_data.edge_label_index,
                              **pre_defined_kwargs)
        
        model = SignNetLinkPredNet(num_feats+K, F_nn[0], F_nn[1]).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        model, train_data_collection, V_collection = train_link_predictor(model, train_data_new, val_data_new, optimizer, 
                                     criterion, n_epochs=n_epochs, K=K, m=m, m2=m2, 
                                     m3=m3, nb_cuts=nb_cuts, remove_isolated=remove_isolated)
        
        test_auc = eval_link_predictor(model, test_data_new)
        results_w_samp_eigs[r] = test_auc
        print(f"Test: {test_auc:.3f}")
        
        print()
        
        # Now with PEs
        
        print("Now with PEs...")
        print()
        
        pre_defined_kwargs = {'eigvecs': V_rec}
        train_data_new = Data(x=train_data.x, edge_index=train_data.edge_index,
                              edge_label=train_data.edge_label,
                              y=train_data.y,edge_label_index=train_data.edge_label_index)
        val_data_new = Data(x=val_data.x, edge_index=val_data.edge_index,
                              edge_label=val_data.edge_label,
                              y=train_data.y,edge_label_index=val_data.edge_label_index,
                              **pre_defined_kwargs)
        pre_defined_kwargs_test = {'eigvecs': V_rec_test}
        test_data_new = Data(x=test_data.x, edge_index=test_data.edge_index,
                              edge_label=test_data.edge_label,
                              y=test_data.y,edge_label_index=test_data.edge_label_index,
                              **pre_defined_kwargs_test)
        
        model = SignNetLinkPredNet(num_feats+F_pe[-1]*K, F_nn[0], F_nn[1], True, 1, F_pe[0], F_pe[1]).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        model, _, _ = train_link_predictor(model, train_data_new, val_data_new, optimizer, 
                                     criterion, n_epochs=n_epochs, K=K, pe=True, m=m, 
                                     m2=m2, m3=m3, nb_cuts=nb_cuts, 
                                     train_data_collection=train_data_collection, 
                                     V_collection=V_collection, remove_isolated=remove_isolated)
        test_auc = eval_link_predictor(model, test_data_new)
        results_w_samp_pe[r] = test_auc
        print(f"Test: {test_auc:.3f}")
        
        print()
    
    ##############################################################################
    ############################# Sampling! ######################################
    ##############################################################################
    
    if do_random_sampl:
    
        print('Sampling at random...')
        print()
        
        sampled_idx2 = list(np.random.choice(np.arange(num_nodes), updated_sz, replace=False))

        # V for train data
        graph_new = train_data.subgraph(torch.tensor(sampled_idx2, device=device, dtype=torch.long))
        
        # Removing isolated nodes
        sampled_idx2_og = sampled_idx2
        if remove_isolated:
            edge_index_new = graph_new.edge_index.clone()
            edge_index_new, _, mask = remove_isolated_nodes(edge_index_new, num_nodes = len(sampled_idx2_og))
            mask = mask.cpu().tolist()
            sampled_idx2 = list(np.array(sampled_idx2_og)[mask])
            graph_new = graph_new.subgraph(torch.tensor(mask, device=device))
        if K > len(sampled_idx2):
            K = len(sampled_idx2)
        len_sampled_idx2[r] = len(sampled_idx2)
        
        graph_new = graph_new.to(device)
        num_nodes_new = graph_new.x.shape[0]
        adj_sparse_new, adj_new = aux_functions.compute_adj_from_data(graph_new)
        
        # Computing normalized Laplacian
        L_new = aux_functions.compute_laplacian(adj_sparse_new, num_nodes_new)
        
        #eigvals_new, V_new = torch.lobpcg(L_new, k=K, largest=False)
        eigvals_new, V_new = torch.linalg.eig(L_new.to_dense())
        eigvals_new = eigvals_new.float()
        V_new = V_new.float()
        idx = torch.argsort(eigvals_new)
        eigvals_new = eigvals_new[idx[0:K]]
        V_new = V_new[:,idx[0:K]]
        V_new = V_new.type(torch.float32)
        V_rec = torch.zeros(num_nodes, K, device=device)
        
        for i in range(V_new.shape[1]):
            v = V_new[:,i]
            V_rec[sampled_idx2,i] = v
            
        # V for test data
        graph_new = test_data.subgraph(torch.tensor(sampled_idx2_og, device=device, dtype=torch.long))
        
        # Removing isolated nodes
        if remove_isolated:
            edge_index_new = graph_new.edge_index.clone()
            edge_index_new, _, mask = remove_isolated_nodes(edge_index_new, num_nodes = len(sampled_idx2_og))
            mask = mask.cpu().tolist()
            sampled_idx2 = list(np.array(sampled_idx2_og)[mask])
            graph_new = graph_new.subgraph(torch.tensor(mask, device=device))
        if K > len(sampled_idx2):
            K = len(sampled_idx2)
        
        graph_new = graph_new.to(device)
        num_nodes_new = graph_new.x.shape[0]
        adj_sparse_new, adj_new = aux_functions.compute_adj_from_data(graph_new)
        
        # Computing normalized Laplacian
        L_new = aux_functions.compute_laplacian(adj_sparse_new, num_nodes_new)
        
        #eigvals_new, V_new = torch.lobpcg(L_new, k=K, largest=False)
        eigvals_new, V_new = torch.linalg.eig(L_new.to_dense())
        eigvals_new = eigvals_new.float()
        V_new = V_new.float()
        idx = torch.argsort(eigvals_new)
        eigvals_new = eigvals_new[idx[0:K]]
        V_new = V_new[:,idx[0:K]]
        V_new = V_new.type(torch.float32)
        V_rec_test = torch.zeros(num_nodes, K, device=device)
        
        for i in range(V_new.shape[1]):
            v = V_new[:,i]
            V_rec_test[sampled_idx2,i] = v
            #rec_error_random[r,i] = torch.linalg.norm(V_rec_test[:,i]-V_test[:,i])/torch.linalg.norm(V_test[:,i])
        
        # Just adding eigenvectors
        
        print("Just adding eigenvectors...")
        print()
        
        pre_defined_kwargs = {'eigvecs': False}
        
        train_data_new = Data(x=train_data.x, edge_index=train_data.edge_index,
                              edge_label=train_data.edge_label,
                              y=train_data.y,edge_label_index=train_data.edge_label_index,
                              **pre_defined_kwargs)
        val_data_new = Data(x=torch.cat((val_data.x,V_rec), dim=1), edge_index=val_data.edge_index,
                              edge_label=val_data.edge_label,
                              y=train_data.y,edge_label_index=val_data.edge_label_index,
                              **pre_defined_kwargs)
        test_data_new = Data(x=torch.cat((test_data.x,V_rec_test), dim=1), edge_index=test_data.edge_index,
                              edge_label=test_data.edge_label,
                              y=test_data.y,edge_label_index=test_data.edge_label_index,
                              **pre_defined_kwargs)
        
        model = SignNetLinkPredNet(num_feats+K, F_nn[0], F_nn[1]).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        model, train_data_collection, V_collection = train_link_predictor(model, train_data_new, val_data_new, optimizer, 
                                     criterion, n_epochs=n_epochs, K=K, m2=m2, m3=m3, remove_isolated=remove_isolated)
        
        test_auc = eval_link_predictor(model, test_data_new)
        results_random_samp_eigs[r] = test_auc
        print(f"Test: {test_auc:.3f}")
        
        print()
        
        # Now with PEs
        
        print("Now with PEs...")
        print()
        
        pre_defined_kwargs = {'eigvecs': V_rec}
        train_data_new = Data(x=train_data.x, edge_index=train_data.edge_index,
                              edge_label=train_data.edge_label,
                              y=train_data.y,edge_label_index=train_data.edge_label_index)
        val_data_new = Data(x=val_data.x, edge_index=val_data.edge_index,
                              edge_label=val_data.edge_label,
                              y=train_data.y,edge_label_index=val_data.edge_label_index,
                              **pre_defined_kwargs)
        pre_defined_kwargs = {'eigvecs': V_rec_test}
        test_data_new = Data(x=test_data.x, edge_index=test_data.edge_index,
                              edge_label=test_data.edge_label,
                              y=test_data.y,edge_label_index=test_data.edge_label_index,
                              **pre_defined_kwargs_test)
        
        model = SignNetLinkPredNet(num_feats+F_pe[-1]*K, F_nn[0], F_nn[1], True, 1, F_pe[0], F_pe[1]).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        model, _, _ = train_link_predictor(model, train_data_new, val_data_new, optimizer, 
                                     criterion, n_epochs=n_epochs, K=K, pe=True, m2=m2, m3=m3,
                                     train_data_collection=train_data_collection,
                                     V_collection=V_collection, remove_isolated=remove_isolated)
        test_auc = eval_link_predictor(model, test_data_new)
        results_random_samp_pe[r] = test_auc
        print(f"Test: {test_auc:.3f}")
        
        print()
 
print('Final results - MAX')
print()

print('Avg. AUC w/o eigenvectors:\t\t\t\t\t%.4f' % np.max(results_no_eigs))
print('Avg. AUC w/ eigenvectors and w/ PEs:\t\t\t\t%.4f    %.4f' % (np.max(results_eigs), np.max(results_pe)))
print('Avg. AUC graphon sampling, idem above:\t\t\t\t%.4f    %.4f' % (np.max(results_w_samp_eigs), np.max(results_w_samp_pe)))
print('Avg. AUC random sampling, idem above:\t\t\t\t%.4f    %.4f' % (np.max(results_random_samp_eigs), np.max(results_random_samp_pe)))
print()    

print('Final results - MEAN')
print()

print('Avg. AUC w/o eigenvectors:\t\t\t\t\t%.4f' % np.mean(results_no_eigs))
print('Avg. AUC w/ eigenvectors and w/ PEs:\t\t\t\t%.4f    %.4f' % (np.mean(results_eigs), np.mean(results_pe)))
print('Avg. AUC graphon sampling, idem above:\t\t\t\t%.4f    %.4f' % (np.mean(results_w_samp_eigs), np.mean(results_w_samp_pe)))
print('Avg. AUC random sampling, idem above:\t\t\t\t%.4f    %.4f' % (np.mean(results_random_samp_eigs), np.mean(results_random_samp_pe)))
print()  

print('Final results - MEDIAN')
print()

print('Avg. AUC w/o eigenvectors:\t\t\t\t\t%.4f' % np.median(results_no_eigs))
print('Avg. AUC w/ eigenvectors and w/ PEs:\t\t\t\t%.4f    %.4f' % (np.median(results_eigs), np.median(results_pe)))
print('Avg. AUC graphon sampling, idem above:\t\t\t\t%.4f    %.4f' % (np.median(results_w_samp_eigs), np.median(results_w_samp_pe)))
print('Avg. AUC random sampling, idem above:\t\t\t\t%.4f    %.4f' % (np.median(results_random_samp_eigs), np.median(results_random_samp_pe)))
print()  

with open(os.path.join(saveDir,'out.txt'), 'w') as f:
    
    print("",file=f)
    
    print('Hyperparameters', file=f)
    print("",file=f)
    
    print('Dataset:\t\t' + data_name, file=f)
    print('Learning rate:\t\t' + str(lr), file=f)
    print('Dataset:\t\t' + data_name, file=f)
    print('Nb. epochs:\t\t' + str(n_epochs), file=f)
    print('Ratio train:\t\t' + str(ratio_train), file=f)
    print('Ratio test:\t\t' + str(ratio_test), file=f)
    print('Nb. realiz.:\t\t' + str(n_realizations), file=f)
    print('Partition sz.:\t\t' + str(m), file=f)
    print('Nb. intervals:\t\t' + str(m2), file=f)
    print('Nb. nodes per int.:\t' + str(m3), file=f)
    print('Nb. comms.:\t\t' + str(nb_cuts), file=f)
    print('F_nn:\t\t\t' + str(F_nn), file=f)
    print('F_pe:\t\t\t' + str(F_pe), file=f)
    print('K:\t\t\t' + str(K), file=f)
    print('Avg. nb. nodes in W samp.:\t' + str(np.mean(len_sampled_idx)), file=f)
    print('Avg. nb. nodes in rand. samp.:\t' + str(np.mean(len_sampled_idx2)), file=f)
    print('Remove isolated:\t\t' + str(remove_isolated), file=f)
    
    print("",file=f)
    
    print('Final results - MAX', file=f)
    print("",file=f)

    print('Avg. AUC w/o eigenvectors:\t\t\t\t\t%.4f' % np.max(results_no_eigs),file=f)
    print('Avg. AUC w/ eigenvectors and w/ PEs:\t\t\t\t%.4f    %.4f' % (np.max(results_eigs), np.max(results_pe)),file=f)
    print('Avg. AUC graphon sampling, idem above:\t\t\t%.4f    %.4f' % (np.max(results_w_samp_eigs), np.max(results_w_samp_pe)),file=f)
    print('Avg. AUC random sampling, idem above:\t\t\t\t%.4f    %.4f' % (np.max(results_random_samp_eigs), np.max(results_random_samp_pe)),file=f)
    print("",file=f)

    print('Final results - MEAN',file=f)
    print("",file=f)

    print('Avg. AUC w/o eigenvectors:\t\t\t\t\t%.4f +/- %.4f' % (np.mean(results_no_eigs),
                                                                 np.std(results_no_eigs)),file=f)
    print('Avg. AUC w/ eigenvectors and w/ PEs:\t\t\t\t%.4f +/- %.4f    %.4f +/- %.4f' % 
          (np.mean(results_eigs), np.std(results_eigs), np.mean(results_pe), np.std(results_pe)),file=f)
    print('Avg. AUC graphon sampling, idem above:\t\t\t%.4f +/- %.4f    %.4f +/- %.4f' % 
          (np.mean(results_w_samp_eigs), np.std(results_w_samp_eigs), np.mean(results_w_samp_pe), 
           np.std(results_w_samp_pe)),file=f)
    print('Avg. AUC random sampling, idem above:\t\t\t\t%.4f +/- %.4f    %.4f +/- %.4f' % 
          (np.mean(results_random_samp_eigs), np.std(results_random_samp_eigs), 
           np.mean(results_random_samp_pe), np.std(results_random_samp_pe)),file=f)
    print("",file=f)

    print('Final results - MEDIAN',file=f)
    print("",file=f)

    print('Avg. AUC w/o eigenvectors:\t\t\t\t\t%.4f' % np.median(results_no_eigs),file=f)
    print('Avg. AUC w/ eigenvectors and w/ PEs:\t\t\t\t%.4f    %.4f' % (np.median(results_eigs), np.median(results_pe)),file=f)
    print('Avg. AUC graphon sampling, idem above:\t\t\t%.4f    %.4f' % (np.median(results_w_samp_eigs), np.median(results_w_samp_pe)),file=f)
    print('Avg. AUC random sampling, idem above:\t\t\t\t%.4f    %.4f' % (np.median(results_random_samp_eigs), np.median(results_random_samp_pe)),file=f)
    print("",file=f)

# Pickling
dict_results = {'results_no_eigs': results_no_eigs,
                'results_eigs': results_eigs,
                'results_pe': results_pe,
                'results_w_samp_eigs': results_w_samp_eigs,
                'results_w_samp_pe': results_w_samp_pe,
                'results_random_samp_eigs': results_random_samp_eigs,
                'results_random_samp_pe': results_random_samp_pe,
                'n_iters': n_iters_per_rlz}
pkl.dump(dict_results, open(os.path.join(saveDir,'results.p'), "wb"))
