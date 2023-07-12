# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:02:23 2023

@author: Luana Ruiz
"""

import numpy as np
import copy

from sklearn.metrics import roc_auc_score

import torch
import time
from tqdm import trange
from torch_geometric.utils import negative_sampling, remove_isolated_nodes
from torch_geometric.loader import DataLoader

from torch_geometric.utils import dropout_edge, to_undirected
import torch_geometric.transforms as T
from torch_geometric.data import Data

from sampling import generate_induced_graphon, greedy, f, sample_clustering
#from reconstruction import f_rec, reconstruct
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
        
        num_nodes = train_data_og_0.x.shape[0]
        device = edge_index.device
        
        train_data_og = Data(x=train_data_og_0.x.clone(), edge_index=edge_index.clone(),
                             y=train_data_og_0.y.clone())
        
        split_collection = []
        if ten_fold:
            num_val = 0.1
        else:
            num_val = 0
        
        for i in range(10):
            eig_edge_index, eig_edge_mask = dropout_edge(edge_index, p=num_val)
            eig_edge_index = to_undirected(eig_edge_index)
            
            data_edge_mask = torch.ones(eig_edge_mask.shape,device=device,dtype=torch.bool)
            data_edge_mask[eig_edge_mask] == 0
            data_edge_index = to_undirected(edge_index[:,data_edge_mask])
            
            split = [eig_edge_index, data_edge_index]
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
                eig_edge_index = split[0]
                eig_data = Data(x=train_data_og.x.clone(), edge_index=eig_edge_index,
                                     y=train_data_og.y.clone())
                
                data_edge_index = split[1]
                train_data = Data(x=train_data_og.x.clone(), edge_index=data_edge_index,
                                     y=train_data_og.y.clone())
                if not ten_fold:
                    train_data = eig_data
                    
                train_data2, _, _ = neg_split(train_data)
                
                train_data_collection.append(train_data2)
                eig_data_collection.append(eig_data)
            
                # V for train data
                adj_sparse, adj = aux_functions.compute_adj_from_data(eig_data)
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
            
            # V for train data
            device = train_data.x.device
            graph_new = train_data.subgraph(torch.tensor(sampled_idx, device=device, dtype=torch.long))
            
            # Removing isolated nodes
            sampled_idx_og = sampled_idx
            edge_index_new = graph_new.edge_index
            edge_index_new, _, mask = remove_isolated_nodes(edge_index_new, num_nodes = len(sampled_idx_og))
            mask = mask.cpu()
            sampled_idx = torch.tensor(sampled_idx_og)[mask==True]
            graph_new = train_data.subgraph(torch.tensor(sampled_idx, device=device, dtype=torch.long))
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
            
            # Removing isolated nodes
            sampled_idx2_og = sampled_idx2
            edge_index_new = graph_new.edge_index
            edge_index_new, _, mask = remove_isolated_nodes(edge_index_new, num_nodes = len(sampled_idx2_og))
            mask = mask.cpu()
            sampled_idx2 = torch.tensor(sampled_idx2_og)[mask==True]
            graph_new = train_data.subgraph(torch.tensor(sampled_idx2, device=device, dtype=torch.long))
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
    
    for epoch in range(n_epochs):
        
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
        
        flag = False
        if val_auc >= best_val_auc:
            best_model = copy.deepcopy(model)
            best_val_auc = val_auc
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")
            flag = True

        if epoch % 10 == 0 and not flag:
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

def train(model, train_data, val_data, optimizer, criterion, batch_size, n_epochs):

    loader = DataLoader([train_data], batch_size=batch_size)  

    # train
    losses = []
    val_accs = []
    best_acc = 0
    best_model = copy.deepcopy(model)
    start = time.time()
    for epoch in trange(n_epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.y
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        losses.append(total_loss)

        if epoch % 1 == 0:
          val_acc = test(model, val_data, is_validation=True)
          val_accs.append(val_acc)
          if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)
        else:
          val_accs.append(val_accs[-1])
        end = time.time()
    final_model = model
    training_time = end-start
    return val_accs, losses, best_model, final_model, best_acc, loader, training_time

def test(test_model, data, is_validation=False, save_model_preds=False):
    
    loader = DataLoader([data])
    test_model.eval()
    correct = 0
    # Note that Cora is only one graph!
    for data in loader:
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = test_model(data).max(dim=1)[1]
            label = data.y

        mask = data.val_mask if is_validation else data.test_mask
        # node classification: only evaluate on nodes in test set
        pred = pred[mask]
        label = label[mask]

        if save_model_preds:
          print ("Saving Model Predictions for Model Type", test_model.type)

          data = {}
          data['pred'] = pred.view(-1).cpu().detach().numpy()
          data['label'] = label.view(-1).cpu().detach().numpy()
            
        correct += pred.eq(label).sum().item()

    total = 0
    for data in loader.dataset:
        total += torch.sum(data.val_mask if is_validation else data.test_mask).item()

    return correct / total