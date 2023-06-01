# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:02:23 2023

@author: Luana Ruiz
"""

import numpy as np
import copy

from sklearn.metrics import roc_auc_score

import torch

from torch_geometric.utils import negative_sampling
import torch_geometric.transforms as T
from torch_geometric.data import Data

from greedy import greedy, f
#from reconstruction import f_rec, reconstruct
from subsampling import sample_clustering
from graphon_sampling import generate_induced_graphon
import aux_functions


zeroTol = 1e-9

def train_link_predictor(model, train_data_og_0, val_data, optimizer, criterion, ten_fold=True,
                         n_epochs=100, K=None, pe=False, m=None, m2=None, m3=None, nb_cuts=None,
                         train_data_collection=None, V_collection=None):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=1)
    
    ########################################################################
    # GENERATE A 10-FOLD SPLIT OF THE DATA FOR THE ENTIRE TRAINING PROCESS #
    ########################################################################
    
    if K is not None:
        # Creating random 10-fold
        edge_index = train_data_og_0.edge_index
        nb_edges = edge_index.shape[1]
        num_nodes = train_data_og_0.x.shape[0]
        device = edge_index.device
        
        train_data_og = Data(x=train_data_og_0.x.clone(), edge_index=edge_index.clone(),
                             y=train_data_og_0.y.clone())
        
        split_collection = []
        if ten_fold:
            num_val = 0.1
        else:
            num_val = 0
        print(train_data_og)
        print(nb_edges)
        nb_data = int(num_val*nb_edges)
        nb_eig = nb_edges-nb_data
        for i in range(10):
            permutation = np.random.permutation(nb_edges)
            split = [permutation[0:nb_eig], permutation[nb_eig:nb_edges]]
            split_collection.append(split)
               
        neg_split = T.RandomLinkSplit(
             num_val=0,
             num_test=0,
             is_undirected=True,
             add_negative_train_samples=False,
             neg_sampling_ratio=1,
        )
    
    best_val_auc = 0
    best_model = None
    
    if K is not None and V_collection is None:
        ###### Eigenvectors
            train_data_collection = []
            eig_data_collection = []
            V_collection = []
        
            for split in split_collection:
                eig_edge_index = edge_index[:,split[0]]
                eig_data = Data(x=train_data_og_0.x.clone(), edge_index=eig_edge_index,
                                     y=train_data_og_0.y.clone())
                
                data_edge_index = edge_index[:,split[1]]
                train_data = Data(x=train_data_og_0.x.clone(), edge_index=data_edge_index,
                                     y=train_data_og_0.y.clone())
                if not ten_fold:
                    train_data = eig_data
                    
                train_data, _, _ = neg_split(train_data)
                
                train_data_collection.append(train_data)
                eig_data_collection.append(eig_data)
            
                # V for train data
                adj_sparse, adj = aux_functions.compute_adj_from_data(eig_data)
                num_nodes = adj.shape[0]
                
                # Computing normalized Laplacian
                L = aux_functions.compute_laplacian(adj_sparse, num_nodes)
                eigvals, V = torch.lobpcg(L, k=K, largest=False)
                #eigvals, V = torch.linalg.eig(L.to_dense())
                #idx = torch.argsort(eigvals)
                #eigvals = L[idx[0:K]]
                #V = V[:,idx[0:K]]
                V_rec = V
                V_collection.append(V_rec)
                
    if m is not None and V_collection is None:
        ###### Graphon sampling
        # Finding sampling set
        n_nodes_per_int, n_nodes_last_int = np.divmod(num_nodes, m)
        k = 5 # k for Anis sampling algorithm
        V_collection = []
        
        for eig_data in eig_data_collection:
            graph_ind = generate_induced_graphon(eig_data, m)
            num_nodes_ind = graph_ind.x.shape[0]
            assert num_nodes_ind == m
            adj_sparse_ind, adj_ind = aux_functions.compute_adj_from_data(graph_ind)
            
            # Computing normalized Laplacian
            L_ind = aux_functions.compute_laplacian(adj_sparse_ind,num_nodes_ind)
            
            lam = eigvals[-1]
            L_aux = L_ind.cpu()
            
            s_vec, n_iters = greedy(f, lam, L_aux, k, m2, exponent=100000000)
            s_vec = torch.tensor(s_vec)
            
            sampled_idx = []
            for i in range(m):
                if s_vec[i] == 1:
                    if i < m-1:
                        cur_adj = adj[i*n_nodes_per_int:(i+1)*n_nodes_per_int,
                                          i*n_nodes_per_int:(i+1)*n_nodes_per_int]
                        idx = sample_clustering(cur_adj, m3, nb_cuts=nb_cuts)#np.random.choice(np.arange(i*n_nodes_per_int,(i+1)*n_nodes_per_int), m3, replace=False)
                    else:
                        if m3 > n_nodes_last_int:
                            m3 = n_nodes_last_int
                        cur_adj = adj[i*n_nodes_per_int:i*n_nodes_per_int+n_nodes_last_int,
                                                    i*n_nodes_per_int:i*n_nodes_per_int+n_nodes_last_int]
                        idx = sample_clustering(cur_adj, m3, nb_cuts=nb_cuts)#np.random.choice(np.arange(i*n_nodes_per_int,
                                                         #i*n_nodes_per_int+n_nodes_last_int), m3, replace=False)
                    idx = np.sort(idx)
                    sampled_idx += list(idx)
            
            # V for train data
            device = train_data.x.device
            graph_new = train_data.subgraph(torch.tensor(sampled_idx, device=device, dtype=torch.long))
            graph_new = graph_new.to(device)
            num_nodes_new = graph_new.x.shape[0]
            adj_sparse_new, adj_new = aux_functions.compute_adj_from_data(graph_new)
            
            # Computing normalized Laplacian
            L_new = aux_functions.compute_laplacian(adj_sparse_new, num_nodes_new)
            
            eigvals_new, V_new = torch.lobpcg(L_new, k=K, largest=False)
            V_new = V_new.type(torch.float32)
            V_rec = torch.zeros(num_nodes, K, device=device)
            
            for i in range(V_new.shape[1]):
                v = V_new[:,i]
                V_rec[sampled_idx,i] = v
            V_collection.append(V_rec)
    
    elif m2 is not None and V_collection is None:
        ###### Random sampling
        sampled_idx2 = list(np.random.choice(np.arange(num_nodes), m2*m3, replace=False))
        V_collection = []

        for eig_data in eig_data_collection:
            # V for train data
            graph_new = eig_data.subgraph(torch.tensor(sampled_idx2, device=device, dtype=torch.long))
            graph_new = graph_new.to(device)
            num_nodes_new = graph_new.x.shape[0]
            adj_sparse_new, adj_new = aux_functions.compute_adj_from_data(graph_new)
            
            # Computing normalized Laplacian
            L_new = aux_functions.compute_laplacian(adj_sparse_new, num_nodes_new)
            
            eigvals_new, V_new = torch.lobpcg(L_new, k=K, largest=False)
            V_new = V_new.type(torch.float32)
            V_rec = torch.zeros(num_nodes, K, device=device)
            
            for i in range(V_new.shape[1]):
                v = V_new[:,i]
                #x0 = np.random.multivariate_normal(np.zeros(num_nodes),np.eye(num_nodes)/np.sqrt(num_nodes))
                #x0[sampled_idx2] = v.cpu().numpy()*np.sqrt(m2*m3)/np.sqrt(num_nodes)
                #v_padded_lb = -torch.ones(num_nodes, device=device)
                #v_padded_lb[sampled_idx2] = v*np.sqrt(m2*m3)/np.sqrt(num_nodes)
                #v_padded_ub = torch.ones(num_nodes, device=device)
                #v_padded_ub[sampled_idx2] = v*np.sqrt(m2*m3)/np.sqrt(num_nodes)
                #V_rec[:,i] = torch.from_numpy(reconstruct(f_rec, x0, v_padded_lb, v_padded_ub, L, k))
                V_rec[sampled_idx2,i] = v
            V_collection.append(V_rec)
        
        
    ##################    
    # START TRAINING #
    ##################
    
    for epoch in range(1, n_epochs + 1):
        
        this_idx = np.random.choice(10)
        
        if m2 is not None:    
            
            pre_defined_kwargs = {'eigvecs': V_collection[this_idx]}
            train_data = train_data_collection[this_idx]
            train_data = Data(x=train_data.x, edge_index=train_data.edge_index,
                                  edge_label=train_data.edge_label,
                                  y=train_data.y,edge_label_index=train_data.edge_label_index,
                                  **pre_defined_kwargs)  
        
        if K is not None:
            
            train_data = train_data_collection[this_idx]
            V_rec = V_collection[this_idx]
            
            if pe is False:
                pre_defined_kwargs = {'eigvecs': False}
                train_data = Data(x=torch.cat((train_data.x,V_rec), dim=1), edge_index=train_data.edge_index,
                                      edge_label=train_data.edge_label,
                                      y=train_data.y,edge_label_index=train_data.edge_label_index,
                                      **pre_defined_kwargs)
            else:
                pre_defined_kwargs = {'eigvecs': V_rec}
                train_data = Data(x=train_data.x, edge_index=train_data.edge_index,
                                      edge_label=train_data.edge_label,
                                      y=train_data.y,edge_label_index=train_data.edge_label_index,
                                      **pre_defined_kwargs)
        else:
            train_data = train_data_og_0

        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index, train_data.eigvecs)

        # sampling training negatives for every training epoch
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label.float())
        loss.backward()
        optimizer.step()

        val_auc = eval_link_predictor(model, val_data)
        if val_auc >= best_val_auc:
            best_model = copy.deepcopy(model)
            best_val_auc = val_auc

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")
            
        scheduler.step()
        
    if best_model is None:
        best_model = model
    return best_model, train_data_collection, V_collection


@torch.no_grad()
def eval_link_predictor(model, data):

    model.eval()
    z = model.encode(data.x, data.edge_index, data.eigvecs)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()

    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy()) 