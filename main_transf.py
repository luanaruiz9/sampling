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

# Save params and results in .txt
# Implement PEs

import sys
import os
import datetime
import pickle as pkl
import numpy as np

import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes

from architecture import GNN
from train_eval import train, test
from sampling import generate_induced_graphon, greedy, f, sample_clustering
import aux_functions

data_name = 'cora'#sys.argv[1]
folder_name = data_name
lr = 0.001#float(sys.argv[2])
n_epochs = 100#int(sys.argv[3])
ratio_train = 0.6#float(sys.argv[4])
ratio_test = 0.2#float(sys.argv[5])
ratio_val = 1-ratio_train-ratio_test
n_realizations = 1#int(sys.argv[6]) #10
m = 100#int(sys.argv[7]) #50 # Number of candidate intervals
m2 = 50#int(sys.argv[8]) #25 # Number of sampled intervals
m3 = 10#int(sys.argv[9]) #3 #8 # How many nodes (points) to sample per sampled interval
nb_cuts = 1#int(sys.argv[10])

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
    
do_no_sampling = True
do_eig = True
do_w_sampl = True
do_random_sampl = True

# Vectors to store test results
results_no_sampl = np.zeros(n_realizations)
results_no_sampl_eig = np.zeros(n_realizations)
results_w_samp = np.zeros(n_realizations)
results_random_samp = np.zeros(n_realizations)
n_iters_per_rlz = np.zeros(n_realizations)

if 'cora' in data_name:
    dataset = Planetoid(root='/tmp/Cora', name='Cora', split='random')
    num_train_per_class = int((ratio_train*dataset[0].num_nodes)/dataset.num_classes)
    num_val = int(ratio_val*dataset[0].num_nodes)
    num_test = dataset[0].num_nodes-num_val-num_train_per_class*dataset.num_classes
    dataset = Planetoid(root='/tmp/Cora', name='Cora', split='random',
                        num_train_per_class=num_train_per_class, num_val=num_val, 
                        num_test=num_test)
elif 'citeseer' in data_name:
    dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer', split='random')
    num_train_per_class = int((ratio_train*dataset[0].num_nodes)/dataset.num_classes)
    num_val = int(ratio_val*dataset[0].num_nodes)
    num_test = dataset[0].num_nodes-num_val-num_train_per_class*dataset.num_classes
    dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer', split='random',
                        num_train_per_class=num_train_per_class, num_val=num_val, 
                        num_test=num_test)
elif 'pubmed' in data_name:
    dataset = Planetoid(root='/tmp/PubMed', name='PubMed', split='random')
    num_train_per_class = int((ratio_train*dataset[0].num_nodes)/dataset.num_classes)
    num_val = int(ratio_val*dataset[0].num_nodes)
    num_test = dataset[0].num_nodes-num_val-num_train_per_class*dataset.num_classes
    dataset = Planetoid(root='/tmp/PubMed', name='PubMed', split='random',
                        num_train_per_class=num_train_per_class, num_val=num_val, 
                        num_test=num_test)

# Sorting nodes by degree
graph_og = dataset[0]
graph_og = graph_og.to(device)
adj_sparse, adj = aux_functions.compute_adj_from_data(graph_og)
num_nodes = graph_og.num_nodes
D = aux_functions.compute_degree(adj_sparse, num_nodes)
deg = torch.diagonal(D.to_dense()).squeeze()
idx = torch.argsort(deg)
idx = idx.to(device)
edge_index = graph_og.edge_index
new_edge_index = torch.zeros(edge_index.shape,dtype=torch.long,device=device)
for i in range(2):
    for j in range(edge_index.shape[1]):
        new_edge_index[i,j] = torch.argwhere(edge_index[i,j]==idx)
graph = Data(x=graph_og.x[idx],edge_index=new_edge_index,y=graph_og.y[idx])

for r in range(n_realizations):
    K = 50
    print('Realization ' + str(r))
    print()
    
    split = T.RandomNodeSplit(
        'random',
        num_train_per_class=num_train_per_class,
        num_val=num_val,
        num_test=num_test
    )

    train_data = split(graph)
    train_data = train_data.to(device)
    val_data = train_data
    test_data = train_data
        
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
        
        if do_eig:
            # Computing normalized Laplacian
            adj_sparse, adj = aux_functions.compute_adj_from_data(graph)
            L = aux_functions.compute_laplacian(adj_sparse, num_nodes)
            eigvals, V = torch.lobpcg(L, k=K, largest=False)
            #eigvals, V = torch.linalg.eig(L.to_dense())
            eigvals = eigvals.float()
            V = V.float()
            idx = torch.argsort(eigvals)
            eigvals = eigvals[idx[0:K]]
            V = V[:,idx[0:K]]
            V = V.to(device)
            
            print('Adding eigenvectors...')
            print()

            train_data_new = Data(x=torch.cat((train_data.x,V), dim=1), edge_index=train_data.edge_index,
                                  y=train_data.y, train_mask=train_data.train_mask,
                                  val_mask=train_data.val_mask, test_mask=train_data.test_mask)
            val_data_new = Data(x=torch.cat((val_data.x,V), dim=1), edge_index=val_data.edge_index,
                                  y=val_data.y, train_mask=val_data.train_mask,
                                  val_mask=val_data.val_mask, test_mask=val_data.test_mask)
            test_data_new = Data(x=torch.cat((test_data.x,V), dim=1), edge_index=test_data.edge_index,
                                  y=test_data.y, train_mask=test_data.train_mask,
                                  val_mask=test_data.val_mask, test_mask=test_data.test_mask)
            
            model = GNN('gcn', [dataset.num_features+K,64,32], [32,dataset.num_classes], softmax=True)
            model = model.to(device)
            optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
            criterion = torch.nn.NLLLoss()
            _,_,model,_,_,_,_ = train(model, train_data_new, val_data_new, optimizer, criterion, 
                          batch_size=1, n_epochs=n_epochs)
            
            test_auc = test(model, test_data_new)
            results_no_sampl_eig[r] = test_auc
            print(f"Test: {test_auc:.3f}")
            
            print()
    
    ##############################################################################
    ############################# Sampling! ######################################
    ##############################################################################
    
    
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
        graph_new = graph.subgraph(torch.tensor(sampled_idx, device=device, dtype=torch.long))
        
        # Removing isolated nodes
        edge_index_new = graph_new.edge_index
        edge_index_new, _, mask = remove_isolated_nodes(edge_index_new, 
                                                                      num_nodes = len(sampled_idx))
        sampled_idx = torch.tensor(sampled_idx)[mask==True]
        graph_new = graph.subgraph(sampled_idx)
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
            #x0 = np.random.multivariate_normal(np.zeros(num_nodes),np.eye(num_nodes)/np.sqrt(num_nodes))
            #x0[sampled_idx] = v.cpu().numpy()*np.sqrt(m2*m3)/np.sqrt(num_nodes)
            #v_padded_lb = -torch.ones(num_nodes, device=device)
            #v_padded_lb[sampled_idx] = v*np.sqrt(m2*m3)/np.sqrt(num_nodes)
            #v_padded_ub = torch.ones(num_nodes, device=device)
            #v_padded_ub[sampled_idx] = v*np.sqrt(m2*m3)/np.sqrt(num_nodes)
            #V_rec[:,i] = torch.from_numpy(reconstruct(f_rec, x0, v_padded_lb, v_padded_ub, L, k))
            V_rec[sampled_idx,i] = v
        
        train_data_new = Data(x=torch.cat((train_data.x,V_rec), dim=1), 
                              edge_index=train_data.edge_index, y=train_data.y,
                              train_mask=val_data.train_mask,
                              val_mask=val_data.val_mask, test_mask=val_data.test_mask)
        val_data_new = Data(x=torch.cat((val_data.x,V_rec), dim=1), 
                              edge_index=val_data.edge_index, y=val_data.y,
                              train_mask=val_data.train_mask,
                              val_mask=val_data.val_mask, test_mask=val_data.test_mask)
        test_data_new = Data(x=torch.cat((test_data.x,V_rec), dim=1), 
                              edge_index=test_data.edge_index, y=test_data.y,
                              train_mask=val_data.train_mask,
                              val_mask=val_data.val_mask, test_mask=val_data.test_mask)
        
        train_data_new = train_data_new.to(device)
        val_data_new = val_data_new.to(device)
        
        in_feats = dataset.num_features
        in_feats += K
        
        model = GNN('gcn', [in_feats,64,32], [32,dataset.num_classes], softmax=True)
        model = model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.NLLLoss()
        _,_,model,_,_,_,_ = train(model, train_data_new, val_data_new, optimizer, criterion, 
                      batch_size=1, n_epochs=n_epochs)
        
        test_auc = test(model, test_data_new)
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
        graph_new = graph.subgraph(torch.tensor(sampled_idx2, device=device, dtype=torch.long))
        
        # Removing isolated nodes
        edge_index_new = graph_new.edge_index
        edge_index_new, _, mask = remove_isolated_nodes(edge_index_new, 
                                                                      num_nodes = len(sampled_idx2))
        sampled_idx2 = torch.tensor(sampled_idx2)[mask==True]
        graph_new = graph.subgraph(sampled_idx2)
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
            #x0[sampled_idx] = v.cpu().numpy()*np.sqrt(m2*m3)/np.sqrt(num_nodes)
            #v_padded_lb = -torch.ones(num_nodes, device=device)
            #v_padded_lb[sampled_idx] = v*np.sqrt(m2*m3)/np.sqrt(num_nodes)
            #v_padded_ub = torch.ones(num_nodes, device=device)
            #v_padded_ub[sampled_idx] = v*np.sqrt(m2*m3)/np.sqrt(num_nodes)
            #V_rec[:,i] = torch.from_numpy(reconstruct(f_rec, x0, v_padded_lb, v_padded_ub, L, k))
            V_rec[sampled_idx2,i] = v
        
        train_data_new = Data(x=torch.cat((train_data.x,V_rec), dim=1), 
                              edge_index=train_data.edge_index, y=train_data.y, 
                              train_mask=val_data.train_mask,
                              val_mask=val_data.val_mask, test_mask=val_data.test_mask)
        val_data_new = Data(x=torch.cat((val_data.x,V_rec), dim=1), 
                              edge_index=val_data.edge_index, y=val_data.y,
                              train_mask=val_data.train_mask,
                              val_mask=val_data.val_mask, test_mask=val_data.test_mask)
        test_data_new = Data(x=torch.cat((test_data.x,V_rec), dim=1), 
                              edge_index=test_data.edge_index, y=test_data.y,
                              train_mask=val_data.train_mask,
                              val_mask=val_data.val_mask, test_mask=val_data.test_mask)
        
        train_data_new = train_data_new.to(device)
        val_data_new = val_data_new.to(device)
        
        in_feats = dataset.num_features
        in_feats += K
        
        model = GNN('gcn', [in_feats,64,32], [32,dataset.num_classes], softmax=True)
        model = model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.NLLLoss()
        _,_,model,_,_,_,_ = train(model, train_data_new, val_data_new, optimizer, criterion, 
                      batch_size=1, n_epochs=n_epochs)
        
        test_auc = test(model, test_data_new)
        results_random_samp[r] = test_auc
        print(f"Test: {test_auc:.3f}")
        
        print()

print('Final results - MAX')
print()

print('Avg. acc. without sampling:\t\t\t\t\t%.4f' % np.max(results_no_sampl))
print('Avg. acc. with eigenvectors:\t\t\t\t\t%.4f' % np.max(results_no_sampl_eig))
print('Avg. acc. idem above, graphon samp.:\t\t\t\t%.4f' % np.max(results_w_samp))
print('Avg. acc. idem above, random samp.:\t\t\t\t%.4f' % np.max(results_random_samp))
print()       

print('Final results - MEAN')
print()

print('Avg. acc. without sampling:\t\t\t\t\t%.4f' % np.mean(results_no_sampl))
print('Avg. acc. with eigenvectors:\t\t\t\t\t%.4f' % np.mean(results_no_sampl_eig))
print('Avg. acc. idem above, graphon samp.:\t\t\t\t%.4f' % np.mean(results_w_samp))
print('Avg. acc. idem above, random samp.:\t\t\t\t%.4f' % np.mean(results_random_samp))
print()    

print('Final results - MEDIAN')
print()

print('Avg. acc. without sampling:\t\t\t\t\t%.4f' % np.median(results_no_sampl))
print('Avg. acc. with eigenvectors:\t\t\t\t\t%.4f' % np.median(results_no_sampl_eig))
print('Avg. acc. idem above, graphon samp.:\t\t\t\t%.4f' % np.median(results_w_samp))
print('Avg. acc. idem above, random samp.:\t\t\t\t%.4f' % np.median(results_random_samp))
print()       

# Pickling
dict_results = {'results_no_sampl': results_no_sampl,
                'results_no_sampl_eig': results_no_sampl_eig,
                'results_w_samp': results_w_samp,
                'results_random_samp': results_random_samp,
                'n_iters': n_iters_per_rlz}
pkl.dump(dict_results, open(os.path.join(saveDir,'results.p'), "wb"))
