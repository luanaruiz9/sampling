# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 17:05:10 2023

@author: Luana Ruiz
"""

import time

import sys
import os
import datetime
import pickle as pkl
import numpy as np

import torch
from torch_geometric.datasets import MalNetTiny
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes, to_networkx, add_self_loops, to_undirected
import networkx as nx

from architecture import GNN
from train_eval import train, test
from sampling import generate_induced_graphon, greedy, f, sample_clustering
#from reconstruction import f_rec, reconstruct
import aux_functions

import random

random.seed(10)

lr = float(sys.argv[1])
n_epochs = int(sys.argv[2])
ratio_train = 0.6
ratio_test = 0.2
ratio_val = 1-ratio_train-ratio_test
n_realizations = int(sys.argv[3]) #10
m = int(sys.argv[4]) #50 # Number of candidate intervals
m2 = int(sys.argv[5]) #25 # Number of sampled intervals
m3 = int(sys.argv[6]) #3 #8 # How many nodes (points) to sample per sampled interval
updated_sz = m2*m3
nb_cuts = int(sys.argv[7])
F_nn = int(sys.argv[8])
F_pe = int(sys.argv[9])
K_in = int(sys.argv[10])
K_og = K_in

thisFilename = 'malnet_' # This is the general name of all related files

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
do_learn_pe = False
do_w_sampl = True
do_random_sampl = True

remove_isolated = False
sort_by_degree = False

dataset_og = MalNetTiny(root='/tmp/MalNetTiny')
dataset = []
y_dict = {}
"""
{3: 53, 0: 173, 4: 54, 1: 132}
"""
for data in dataset_og:
    if data.num_nodes >= 4500:
        label = data.y.cpu().numpy()[0]
        if label not in y_dict.keys():
            y_dict[label] = 1
        else:
            y_dict[label] += 1
        if label > 1:
            data.y -= 1
        if y_dict[label] <= 54: #54
            dataset.append(data)
        
print("length of dataset ", len(dataset))

transformed_dataset = []
transform = T.ToUndirected()
pre_defined_kwargs = {'eigvecs': False}
num_feats = 1
num_classes = 4#dataset_og.num_classes

for graph_og in dataset:
    
    graph_og = transform(graph_og)
    #graph_og = graph_og.subgraph(torch.arange(500)) # comment it out
    
    # Adding all-ones input features
    graph_og = Data(x=torch.ones(graph_og.num_nodes,1),edge_index=graph_og.edge_index,y=graph_og.y)
    
    # Sorting nodes by degree
    if sort_by_degree:
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
        graph_og = Data(x=graph_og.x,edge_index=new_edge_index,y=graph_og.y)

    graph = Data(x=graph_og.x, edge_index=graph_og.edge_index, 
                 edge_weight=graph_og.edge_weight, y=graph_og.y,**pre_defined_kwargs)
    graph = graph.to(device)
    transformed_dataset.append(graph)

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

n_total = len(dataset)
n_train = int(n_total*ratio_train)
n_test = int(n_total*ratio_test)
n_val = n_total-n_train-n_test

all_data = transformed_dataset
data_exists = False

for r in range(n_realizations):
    
    K = K_in
    data_exists = False
    print('Realization ' + str(r))
    print() 
    
    random_permutation = np.random.permutation(n_total)
    train_idx = list(random_permutation[0:n_train])
    test_idx = list(random_permutation[n_train:n_train+n_test])
    val_idx = list(random_permutation[-n_val:])
    
    if do_no_pe:
        
        train_data = [all_data[i] for i in train_idx]
        val_data = [all_data[i] for i in val_idx]
        test_data = [all_data[i] for i in test_idx]
        
        model = GNN('gcn', [num_feats,F_nn,F_nn,F_nn,F_nn], [], softmax=False, aggregate=True, 
                    num_graph_classes = num_classes)
        model = model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.NLLLoss()
        start = time.time()
        _,_,model,_,_,_,_ = train(model, train_data, val_data, optimizer, criterion, 
                      batch_size=8, n_epochs=n_epochs)
        end = time.time()
        print('Elapsed time: ' + str(end-start))
        print()
        test_auc = test(model, test_data)
        results_no_eigs[r] = test_auc
        print(f"Test: {test_auc:.3f}")
        
        print()
  
    ##############################################################################
    ######################## Adding eigenvectors #################################
    ##############################################################################
    
    eigenvector_time = 0
    if do_eig:
    
        print('Adding eigenvectors...')
        print()
        
        pre_defined_kwargs = {'eigvecs': False}
    
        all_data_new = []
        all_Vs= []
        
        for data_elt in all_data:
            # V for train data
            adj_sparse, adj = aux_functions.compute_adj_from_data(data_elt)
            num_nodes = adj.shape[0]
            
            # Computing normalized Laplacian
            L = aux_functions.compute_laplacian(adj_sparse, num_nodes)
            start = time.time()
            eigvals, V = torch.lobpcg(L, k=K, largest=False)
            end = time.time()
            eigenvector_time += end-start
            #eigvals, V = torch.linalg.eig(L.to_dense())
            eigvals = torch.abs(eigvals).float()
            V = V.float()
            idx = torch.argsort(eigvals)
            eigvals = eigvals[idx[0:K]]
            V = V[:,idx[0:K]]
            all_Vs.append(V)
            
            data_elt_new = Data(x=torch.cat((data_elt.x,V), dim=1), 
                                      edge_index=data_elt.edge_index,
                                      y=data_elt.y,
                                      **pre_defined_kwargs)
            all_data_new.append(data_elt_new)
            
        print('Eigenvector time: ' + str(eigenvector_time))
        print()
        
        train_data_new = [all_data_new[i] for i in train_idx]
        val_data_new = [all_data_new[i] for i in val_idx]
        test_data_new = [all_data_new[i] for i in test_idx]
        
        model = GNN('gcn', [num_feats+K,F_nn,F_nn,F_nn,F_nn], [], softmax=False, aggregate=True, 
                    num_graph_classes = num_classes)
        model = model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        start = time.time()
        _,_,model,_,_,_,_ = train(model, train_data_new, val_data_new, optimizer, criterion, 
                      batch_size=8, n_epochs=n_epochs)
        end = time.time()
        print('Elapsed time: ' + str(end-start))
        print()
        test_auc = test(model, test_data_new)
        results_eigs[r] = test_auc
        print(f"Test: {test_auc:.3f}")
        
        print()

    """
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
                                         criterion, n_epochs=n_epochs, K=K, pe=True, ten_fold=ten_fold)
            test_auc = eval_link_predictor(model, test_data_new)
            results_pe[r] = test_auc
            print(f"Test: {test_auc:.3f}")
            
            print()
     """
   
    ##############################################################################
    ############################# Sampling! ######################################
    ##############################################################################
    
    w_eigenvector_time = 0
    if do_w_sampl:
        
        print('Sampling with spectral proxies...')
        print()
        
        # Just adding eigenvectors
        
        print("Just adding eigenvectors...")
        print()
        
        all_data_new =[]
        all_Vs_w = []
        
        if os.path.isfile("graphon_data"+str(r)+".p"):
            data_exists = True
            all_data_new = pkl.load(open("graphon_data"+str(r)+".p","rb"))
        else:
            all_data_new = []
        
            # Train data
            for data_elt in all_data:
                K = K_og
                # Finding sampling set
                num_nodes = data_elt.x.shape[0]
                n_nodes_per_int, n_nodes_last_int = np.divmod(num_nodes, m)
                graph_ind = generate_induced_graphon(data_elt, m)
                num_nodes_ind = graph_ind.x.shape[0]
                assert num_nodes_ind == m
                adj_sparse_ind, adj_ind = aux_functions.compute_adj_from_data(graph_ind)
                
                # Computing normalized Laplacian
                L_ind = aux_functions.compute_laplacian(adj_sparse_ind,num_nodes_ind)
                
                lam = eigvals[-1]
                L_aux = L_ind.cpu()
                k = 5
                
                s_vec, n_iters = greedy(f, lam, L_aux, k, m2)
                    
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
                
                # V for train data
                graph_new = data_elt.clone().subgraph(torch.tensor(sampled_idx, device=device, dtype=torch.long))
        
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
                start = time.time()
                eigvals_new, V_new = torch.linalg.eig(L_new.to_dense())
                end = time.time()
                w_eigenvector_time += end-start
                
                eigvals_new = torch.abs(eigvals_new).float()
                V_new = V_new.float()
                idx = torch.argsort(eigvals_new)
                eigvals_new = eigvals_new[idx[0:K]]
                
                V_new = V_new[:,idx[0:K]]
                V_new = V_new.type(torch.float32)
                V_rec = torch.zeros(num_nodes, K_og, device=device)
                
                for i in range(V_new.shape[1]):
                    v = V_new[:,i]
                    V_rec[sampled_idx,i] = v
                    
                all_Vs_w.append(V_rec)
                
                pre_defined_kwargs = {'eigvecs': False}
                
                data_elt_new = Data(x=torch.cat((data_elt.x,V_rec), dim=1),
                                          edge_index=data_elt.edge_index,
                                          y=data_elt.y,
                                          **pre_defined_kwargs)
                all_data_new.append(data_elt_new)
        
        print('Eigenvector time: ' + str(w_eigenvector_time))
        print()
        
        train_data_new = [all_data_new[i] for i in train_idx]
        val_data_new = [all_data_new[i] for i in val_idx]
        test_data_new = [all_data_new[i] for i in test_idx]
        
        model = GNN('gcn', [num_feats+K,F_nn,F_nn,F_nn,F_nn], [], softmax=False, aggregate=True, 
                    num_graph_classes = num_classes)
        model = model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        start = time.time()
        _,_,model,_,_,_,_ = train(model, train_data_new, val_data_new, optimizer, criterion, 
                      batch_size=8, n_epochs=n_epochs)
        end = time.time()
        print('Elapsed time: ' + str(end-start))
        print()
        test_auc = test(model, test_data_new)
        results_w_samp_eigs[r] = test_auc
        print(f"Test: {test_auc:.3f}")
        
        print()
        
        if data_exists == False:
            pkl.dump(all_data_new,open("graphon_data"+str(r)+".p","wb"))
        
     ##############################################################################
     ############################# Sampling! ######################################
     ##############################################################################
     
    r_eigenvector_time= 0
    if do_random_sampl:
     
         print('Sampling at random...')
         print()

         # Just adding eigenvectors
         
         print("Just adding eigenvectors...")
         print()
         
         all_Vs_r = []
         
         all_data_new = []
         
         # Train data
         for data_elt in all_data:
             K = K_og
             # Finding sampling set
             num_nodes = data_elt.x.shape[0]
             sampled_idx = list(np.random.choice(np.arange(num_nodes), updated_sz, replace=False))
             
             # V for train data
             graph_new = data_elt.clone().subgraph(torch.tensor(sampled_idx, device=device, dtype=torch.long))
     
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
             len_sampled_idx2[r] = len(sampled_idx)
     
             graph_new = graph_new.to(device)
             num_nodes_new = graph_new.x.shape[0]
             adj_sparse_new, adj_new = aux_functions.compute_adj_from_data(graph_new)
             
             # Computing normalized Laplacian
             L_new = aux_functions.compute_laplacian(adj_sparse_new, num_nodes_new)
             
             #eigvals_new, V_new = torch.lobpcg(L_new, k=K, largest=False)
             start = time.time()
             eigvals_new, V_new = torch.linalg.eig(L_new.to_dense())
             end = time.time()
             end = time.time()
             r_eigenvector_time += end-start
             
             eigvals_new = torch.abs(eigvals_new).float()
             V_new = V_new.float()
             idx = torch.argsort(eigvals_new)
             eigvals_new = eigvals_new[idx[0:K]]
             
             V_new = V_new[:,idx[0:K]]
             V_new = V_new.type(torch.float32)
             V_rec = torch.zeros(num_nodes, K_og, device=device)
             
             for i in range(V_new.shape[1]):
                 v = V_new[:,i]
                 V_rec[sampled_idx,i] = v
                 
             all_Vs_r.append(V_rec)
             
             pre_defined_kwargs = {'eigvecs': False}
             
             data_elt_new = Data(x=torch.cat((data_elt.x,V_rec), dim=1),
                                       edge_index=data_elt.edge_index,
                                       y=data_elt.y,
                                       **pre_defined_kwargs)
             all_data_new.append(data_elt_new)
         
         print('Eigenvector time: ' + str(r_eigenvector_time))
         print()
            
         train_data_new = [all_data_new[i] for i in train_idx]
         val_data_new = [all_data_new[i] for i in val_idx]
         test_data_new = [all_data_new[i] for i in test_idx]
         
         model = GNN('gcn', [num_feats+K,F_nn,F_nn,F_nn,F_nn], [], softmax=False, aggregate=True, 
                     num_graph_classes = num_classes)
         model = model.to(device)
         optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
         criterion = torch.nn.CrossEntropyLoss()
         start = time.time()
         _,_,model,_,_,_,_ = train(model, train_data_new, val_data_new, optimizer, criterion, 
                       batch_size=8, n_epochs=n_epochs)
         end = time.time()
         print('Elapsed time: ' + str(end-start))
         print()
         test_auc = test(model, test_data_new)
         results_random_samp_eigs[r] = test_auc
         print(f"Test: {test_auc:.3f}")
         
         print()
        
""""        
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
                                     V_collection=V_collection, remove_isolated=remove_isolated, 
                                     ten_fold=ten_fold)
        test_auc = eval_link_predictor(model, test_data_new)
        results_w_samp_pe[r] = test_auc
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
                                     V_collection=V_collection, remove_isolated=remove_isolated,
                                     ten_fold=ten_fold)
        test_auc = eval_link_predictor(model, test_data_new)
        results_random_samp_pe[r] = test_auc
        print(f"Test: {test_auc:.3f}")
        
        print()

"""
    
print('Final results - MAX')
print()

print('AUC w/o eigenvectors:\t\t\t\t\t\t%.4f' % np.max(results_no_eigs))
print('AUC w/ eigenvectors and w/ PEs:\t\t\t\t\t%.4f    %.4f' % (np.max(results_eigs), np.max(results_pe)))
print('AUC graphon sampling, idem above:\t\t\t\t%.4f    %.4f' % (np.max(results_w_samp_eigs), np.max(results_w_samp_pe)))
print('AUC random sampling, idem above:\t\t\t\t%.4f    %.4f' % (np.max(results_random_samp_eigs), np.max(results_random_samp_pe)))
print()    

print('Final results - MEAN')
print()

print('AUC w/o eigenvectors:\t\t\t\t\t\t%.4f' % np.mean(results_no_eigs))
print('AUC w/ eigenvectors and w/ PEs:\t\t\t\t\t%.4f    %.4f' % (np.mean(results_eigs), np.mean(results_pe)))
print('AUC graphon sampling, idem above:\t\t\t\t%.4f    %.4f' % (np.mean(results_w_samp_eigs), np.mean(results_w_samp_pe)))
print('AUC random sampling, idem above:\t\t\t\t%.4f    %.4f' % (np.mean(results_random_samp_eigs), np.mean(results_random_samp_pe)))
print()  

print('Final results - MEDIAN')
print()

print('AUC w/o eigenvectors:\t\t\t\t\t\t%.4f' % np.median(results_no_eigs))
print('AUC w/ eigenvectors and w/ PEs:\t\t\t\t\t%.4f    %.4f' % (np.median(results_eigs), np.median(results_pe)))
print('AUC graphon sampling, idem above:\t\t\t\t%.4f    %.4f' % (np.median(results_w_samp_eigs), np.median(results_w_samp_pe)))
print('AUC random sampling, idem above:\t\t\t\t%.4f    %.4f' % (np.median(results_random_samp_eigs), np.median(results_random_samp_pe)))
print()  

with open(os.path.join(saveDir,'out.txt'), 'w') as f:
    
    print("",file=f)
    
    print('Hyperparameters', file=f)
    print("",file=f)
    
    print('Dataset:\t\tMalNet Tiny', file=f)
    print('Learning rate:\t\t' + str(lr), file=f)
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

    print('AUC w/o eigenvectors:\t\t\t\t\t\t%.4f' % np.max(results_no_eigs),file=f)
    print('AUC w/ eigenvectors and w/ PEs:\t\t\t\t\t%.4f    %.4f' % (np.max(results_eigs), np.max(results_pe)),file=f)
    print('AUC graphon sampling, idem above:\t\t\t%.4f    %.4f' % (np.max(results_w_samp_eigs), np.max(results_w_samp_pe)),file=f)
    print('AUC random sampling, idem above:\t\t\t\t%.4f    %.4f' % (np.max(results_random_samp_eigs), np.max(results_random_samp_pe)),file=f)
    print("",file=f)

    print('Final results - MEAN',file=f)
    print("",file=f)

    print('AUC w/o eigenvectors:\t\t\t\t\t\t%.4f +/- %.4f' % (np.mean(results_no_eigs),
                                                                 np.std(results_no_eigs)),file=f)
    print('AUC w/ eigenvectors and w/ PEs:\t\t\t\t\t%.4f +/- %.4f    %.4f +/- %.4f' % 
          (np.mean(results_eigs), np.std(results_eigs), np.mean(results_pe), np.std(results_pe)),file=f)
    print('AUC graphon sampling, idem above:\t\t\t%.4f +/- %.4f    %.4f +/- %.4f' % 
          (np.mean(results_w_samp_eigs), np.std(results_w_samp_eigs), np.mean(results_w_samp_pe), 
           np.std(results_w_samp_pe)),file=f)
    print('AUC random sampling, idem above:\t\t\t\t%.4f +/- %.4f    %.4f +/- %.4f' % 
          (np.mean(results_random_samp_eigs), np.std(results_random_samp_eigs), 
           np.mean(results_random_samp_pe), np.std(results_random_samp_pe)),file=f)
    print("",file=f)

    print('Final results - MEDIAN',file=f)
    print("",file=f)

    print('AUC w/o eigenvectors:\t\t\t\t\t\t%.4f' % np.median(results_no_eigs),file=f)
    print('AUC w/ eigenvectors and w/ PEs:\t\t\t\t\t%.4f    %.4f' % (np.median(results_eigs), np.median(results_pe)),file=f)
    print('AUC graphon sampling, idem above:\t\t\t%.4f    %.4f' % (np.median(results_w_samp_eigs), np.median(results_w_samp_pe)),file=f)
    print('AUC random sampling, idem above:\t\t\t\t%.4f    %.4f' % (np.median(results_random_samp_eigs), np.median(results_random_samp_pe)),file=f)
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