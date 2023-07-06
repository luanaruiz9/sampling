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
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data

from architecture import GNN
from train_eval import train, test
from greedy import greedy, f
from subsampling import sample_clustering
from graphon_sampling import generate_induced_graphon
import aux_functions

data_name = sys.argv[1]
folder_name = data_name
lr = float(sys.argv[2])
n_epochs = int(sys.argv[3])
n_realizations = int(sys.argv[4]) #10
m = int(sys.argv[5]) #100 # Number of candidate intervals
m2 = int(sys.argv[6]) #10 # Number of sampled intervals
m3 = int(sys.argv[7]) #10 # How many nodes (points) to sample per sampled interval
nb_cuts = int(sys.argv[8])

thisFilename = folder_name + '_cora' # This is the general name of all related files

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
    
K = 20
do_no_sampling = True
do_w_sampl = True
do_random_sampl = True

dataset = Planetoid(root='/tmp/Cora', name='Cora')
graph_og = dataset[0]
#graph_og = graph_og.subgraph(torch.arange(500)) # comment it out
pre_defined_kwargs = {'eigvecs': False}
graph = Data(x=graph_og.x, edge_index=graph_og.edge_index, 
             edge_weight=graph_og.edge_weight, y=graph_og.y,**pre_defined_kwargs)
graph = graph.to(device)

# Vectors to store test results
results_no_sampl = np.zeros(n_realizations)
results_w_samp = np.zeros(n_realizations)
results_random_samp = np.zeros(n_realizations)
n_iters_per_rlz = np.zeros(n_realizations)

for r in range(n_realizations):
    
    print('Realization ' + str(r))
    print()
    
    if 'cora' in data_name:
        dataset = Planetoid(root='/tmp/Cora', name='Cora', split='full')
    elif 'citeseer' in data_name:
        dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer', split='full')
    elif 'pubmed' in data_name:
        dataset = Planetoid(root='/tmp/PubMed', name='PubMed', split='full')
    
    # Computing normalized Laplacian
    graph_og = dataset[0]

    # Sorting nodes by degree
    adj_sparse, adj = aux_functions.compute_adj_from_data(graph_og)
    num_nodes = adj.shape[0]

    L = aux_functions.compute_laplacian(adj_sparse, num_nodes)
    eigvals, V = torch.lobpcg(L, k=K, largest=False)
    
    train_data = dataset.get('train')
    train_data = train_data.to(device)
    val_data = dataset.get('val')
    val_data = val_data.to(device)
    test_data = dataset.get('test')
    test_data = test_data.to(device)
    
    if do_no_sampling:
    
        model = GNN('gcn', [dataset.num_features,64,32], [32,dataset.num_classes], softmax=True)
        model = model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.NLLLoss()
        _,_,model,_,_,_,_ = train(model, train_data, val_data, optimizer, criterion, 
                      batch_size=1, n_epochs=n_epochs)
        
        test_auc = test(model, test_data)
        results_no_sampl[r] = test_auc
        print(f"Test: {test_auc:.3f}")
        
        print()
    
    ##############################################################################
    ############################# Sampling! ######################################
    ##############################################################################
    D = aux_functions.compute_degree(adj_sparse, num_nodes)
    deg = torch.diagonal(D.to_dense()).squeeze()
    idx = torch.argsort(deg)
    idx = idx.to(device)
    train_data = train_data.subgraph(idx)
    val_data = val_data.subgraph(idx)
    test_data = test_data.subgraph(idx)
    
    if do_w_sampl:
    
        print('Sampling with spectral proxies...')
        print()
        
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
        train_data_new = train_data.subgraph(torch.tensor(sampled_idx, device=device, dtype=torch.long))
        train_data_new = train_data_new.to(device)
        num_nodes_new = train_data_new.x.shape[0]
        val_data_new = val_data.subgraph(torch.tensor(sampled_idx, device=device, dtype=torch.long))
        val_data_new = val_data_new.to(device)
        
        model = GNN('gcn', [dataset.num_features,64,32], [32,dataset.num_classes], softmax=True)
        model = model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.NLLLoss()
        _,_,model,_,_,_,_ = train(model, train_data_new, val_data_new, optimizer, criterion, 
                      batch_size=1, n_epochs=n_epochs)
        
        test_auc = test(model, test_data)
        results_w_samp[r] = test_auc
        print(f"Test: {test_auc:.3f}")
        
        print()
    
    ##############################################################################
    ############################# Sampling! ######################################
    ##############################################################################
    
    if do_random_sampl:
    
        print('Sampling at random...')
        print()
        
        sampled_idx2 = list(np.random.choice(np.arange(num_nodes), m2*m3, replace=False))

         # V for train data
        train_data_new = train_data.subgraph(torch.tensor(sampled_idx, device=device, dtype=torch.long))
        train_data_new = train_data_new.to(device)
        num_nodes_new = train_data_new.x.shape[0]
        val_data_new = val_data.subgraph(torch.tensor(sampled_idx, device=device, dtype=torch.long))
        val_data_new = val_data_new.to(device)
        
        model = GNN('gcn', [dataset.num_features,64,32], [32,dataset.num_classes], softmax=True)
        model = model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.NLLLoss()
        _,_,model,_,_,_,_ = train(model, train_data_new, val_data_new, optimizer, criterion, 
                      batch_size=1, n_epochs=n_epochs)
        
        test_auc = test(model, test_data)
        results_random_samp[r] = test_auc
        print(f"Test: {test_auc:.3f}")
        
        print()
 
print('Final results')
print()

print('Avg. acc. w/o sampling:\t\t\t\t\t%.4f' % np.mean(results_no_sampl))
print('Avg. acc. graphon sampling:\t\t\t\t%.4f' % np.mean(results_w_samp))
print('Avg. acc. random sampling:\t\t\t\t%.4f' % np.mean(results_random_samp))
print()    

# Pickling
dict_results = {'results_no_sampl': results_no_sampl,
                'results_w_samp': results_w_samp,
                'results_random_samp': results_random_samp,
                'n_iters': n_iters_per_rlz}
pkl.dump(dict_results, open(os.path.join(saveDir,'results.p'), "wb"))
