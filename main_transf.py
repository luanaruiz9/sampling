# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:37:09 2023

@author: Luana Ruiz
"""

import sys
import os
import datetime
import pickle as pkl
import numpy as np

import torch
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import remove_isolated_nodes, index_to_mask


from architecture import GNN
from train_eval import train, test
from sampling import generate_induced_graphon, greedy, f, sample_clustering
import aux_functions

# SBM - graphon case
# Learning by transference?

data_name = sys.argv[1]
lr = float(sys.argv[2]) # 0.001
n_epochs = int(sys.argv[3]) #100
ratio_train = 0.6
ratio_test = 0.2
ratio_val = 1-ratio_train-ratio_test
n_realizations = int(sys.argv[4]) #10
m = int(sys.argv[5]) #20 # Number of candidate intervals
m2 = int(sys.argv[6]) #10 # Number of sampled intervals
m3 = int(sys.argv[7]) #10 # How many nodes (points) to sample per sampled interval
updated_sz = m2*m3
nb_cuts = int(sys.argv[8]) #1
F_nn = int(sys.argv[9]) #32
K_in = int(sys.argv[10]) #7

thisFilename = 'transf_' + data_name # This is the general name of all related files

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
do_learn_pe = True
do_w_sampl = True
do_random_sampl = True

remove_isolated = False
sort_by_degree = True

# Vectors to store test results
results_no_sampl = np.zeros(n_realizations)
results_no_sampl_eig = np.zeros(n_realizations)
results_w_samp = np.zeros(n_realizations)
results_random_samp = np.zeros(n_realizations)
n_iters_per_rlz = np.zeros(n_realizations)
len_sampled_idx = np.zeros(n_realizations)
len_sampled_idx2 = np.zeros(n_realizations)

in_feats = 0
C = 0

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
elif 'ogb' in data_name:
    sort_by_degree = False
    dataset = PygNodePropPredDataset(name='ogbn-mag')
    rel_data = dataset[0]

    split_idx = dataset.get_idx_split()

    train_idx = split_idx['train']['paper']
    val_idx = split_idx['valid']['paper']
    test_idx = split_idx['test']['paper']

    nTrain = torch.sum(train_idx).item()
    nVal = torch.sum(val_idx).item()
    nTest = torch.sum(test_idx).item()

    m = rel_data.x_dict['paper'].shape[0]

    # We are only interested in paper <-> paper relations.
    data = Data(
        x=rel_data.x_dict['paper'],
        edge_index=rel_data.edge_index_dict[('paper', 'cites', 'paper')],
        y=rel_data.y_dict['paper'],
        train_mask=index_to_mask(train_idx,size=m),
        val_mask=index_to_mask(val_idx,size=m),
        test_mask=index_to_mask(test_idx,size=m))

    data = T.ToUndirected()(data)
    data = data.subgraph(torch.randperm(m)[0:50000]) # Restricting to 200k 
                                                    # nodes due to memory limitations
    in_feats = rel_data.x_dict['paper'].shape[1]
    C = dataset.num_classes
    dataset = [data]

if in_feats == 0:
    in_feats = dataset.num_features
if C == 0:
    C = dataset.num_classes

graph_og = dataset[0]
transform = T.ToUndirected()
graph_og = transform(graph_og)
graph_og = graph_og.to(device)
num_nodes = graph_og.num_nodes
if sort_by_degree:
    # Sorting nodes by degree
    adj_sparse, adj = aux_functions.compute_adj_from_data(graph_og)
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
else:
    graph = graph_og.clone()

for r in range(n_realizations):
    K = K_in
    
    print('Realization ' + str(r))
    print()
    
    if 'ogb' in data_name:
        train_data = graph
        val_data = graph
        test_data = graph
    else:
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
        
        model = GNN('gcn', [in_feats,F_nn,F_nn], [F_nn,C], softmax=True)
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
    
    # Computing normalized Laplacian
    adj_sparse, adj = aux_functions.compute_adj_from_data(graph)
    L = aux_functions.compute_laplacian(adj_sparse, num_nodes)
    eigvals, V = torch.lobpcg(L, k=K, largest=False)
    #eigvals, V = torch.linalg.eig(L.to_dense())
    eigvals = torch.abs(eigvals).float()
    V = V.float()
    idx = torch.argsort(eigvals)
    eigvals = eigvals[idx[0:K]]
    V = V[:,idx[0:K]]
    
    if do_w_sampl:
    
        print('Sampling with spectral proxies...')
        print()
        
        if True:# r == 0: # Only sample intervals once
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
        
        # Train data
        train_data_new = train_data.subgraph(torch.tensor(sampled_idx, device=device, dtype=torch.long))
        val_data_new = val_data.subgraph(torch.tensor(sampled_idx, device=device, dtype=torch.long))

        # Removing isolated nodes
        sampled_idx_og = sampled_idx
        if remove_isolated:
            edge_index_new = train_data_new.edge_index.clone()
            edge_index_new, _, mask = remove_isolated_nodes(edge_index_new, num_nodes = len(sampled_idx_og))
            mask = mask.cpu().tolist()
            sampled_idx = list(np.array(sampled_idx_og)[mask])
            train_data_new = train_data_new.subgraph(torch.tensor(mask, device=device))
            val_data_new = val_data_new.subgraph(torch.tensor(mask, device=device))
        if K > len(sampled_idx):
            K = len(sampled_idx)
        len_sampled_idx[r] = len(sampled_idx)

        train_data_new = train_data_new.to(device)
        val_data_new = val_data_new.to(device)
                
        model = GNN('gcn', [in_feats,F_nn,F_nn], [F_nn,C], softmax=True)
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
        
        sampled_idx2 = list(np.random.choice(np.arange(num_nodes), updated_sz, replace=False))

        # Train data
        train_data_new = train_data.subgraph(torch.tensor(sampled_idx2, device=device, dtype=torch.long))
        val_data_new = val_data.subgraph(torch.tensor(sampled_idx2, device=device, dtype=torch.long))

        # Removing isolated nodes
        sampled_idx2_og = sampled_idx2
        if remove_isolated:
            edge_index_new = train_data_new.edge_index.clone()
            edge_index_new, _, mask = remove_isolated_nodes(edge_index_new, num_nodes = len(sampled_idx2_og))
            mask = mask.cpu().tolist()
            sampled_idx2 = list(np.array(sampled_idx2_og)[mask])
            train_data_new = train_data_new.subgraph(torch.tensor(mask, device=device))
            val_data_new = val_data_new.subgraph(torch.tensor(mask, device=device))
        if K > len(sampled_idx2):
            K = len(sampled_idx2)
        len_sampled_idx2[r] = len(sampled_idx2)

        train_data_new = train_data_new.to(device)
        val_data_new = val_data_new.to(device)
        
        model = GNN('gcn', [in_feats,F_nn,F_nn], [F_nn,C], softmax=True)
        model = model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.NLLLoss()
        _,_,model,_,_,_,_ = train(model, train_data_new, val_data_new, optimizer, criterion, 
                      batch_size=1, n_epochs=n_epochs)
        
        test_auc = test(model, test_data)
        results_random_samp[r] = test_auc
        print(f"Test: {test_auc:.3f}")
        
        print()

print('Final results - MAX')
print()

print('Acc. without sampling:\t\t\t\t%.4f' % np.max(results_no_sampl))
print('Acc. graphon sampling:\t\t\t\t%.4f' % np.max(results_w_samp))
print('Acc. random sampling:\t\t\t\t%.4f' % np.max(results_random_samp))
print()       

print('Final results - MEAN')
print()

print('Acc. without sampling:\t\t\t\t%.4f' % np.mean(results_no_sampl))
print('Acc. graphon sampling:\t\t\t\t%.4f' % np.mean(results_w_samp))
print('Acc. random sampling:\t\t\t\t%.4f' % np.mean(results_random_samp))
print()    

print('Final results - MEDIAN')
print()

print('Acc. without sampling:\t\t\t\t%.4f' % np.median(results_no_sampl))
print('Acc. graphon sampling:\t\t\t\t%.4f' % np.median(results_w_samp))
print('Acc. random sampling:\t\t\t\t%.4f' % np.median(results_random_samp))
print()       

with open(os.path.join(saveDir,'out.txt'), 'w') as f:
    
    print("",file=f)
    
    print('Hyperparameters', file=f)
    print("",file=f)
    
    print('Dataset:\t\t' + data_name, file=f)
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
    print('K:\t\t\t' + str(K), file=f)
    print('Avg. nb. nodes in W samp.:\t' + str(np.mean(len_sampled_idx)), file=f)
    print('Avg. nb. nodes in rand. samp.:\t' + str(np.mean(len_sampled_idx2)), file=f)
    print('Remove isolated:\t\t' + str(remove_isolated), file=f)
    
    print("",file=f)
    
    print('Final results - MAX', file=f)
    print("",file=f)

    print('Acc. without sampling:\t\t\t\t%.4f' % np.max(results_no_sampl),file=f)
    print('Acc. graphon sampling:\t\t\t%.4f' % (np.max(results_w_samp)),file=f)
    print('Acc. random sampling:\t\t\t\t%.4f' % (np.max(results_random_samp)),file=f)
    print("",file=f)

    print('Final results - MEAN',file=f)
    print("",file=f)

    print('Acc. without sampling:\t\t\t\t%.4f +/- %.4f' % (np.mean(results_no_sampl),
                                                                 np.std(results_no_sampl)),file=f)
    print('Acc. graphon sampling:\t\t\t%.4f +/- %.4f' % 
          (np.mean(results_w_samp), np.std(results_w_samp)),file=f)
    print('Acc. random sampling:\t\t\t\t%.4f +/- %.4f' % 
          (np.mean(results_random_samp), np.std(results_random_samp)),file=f)
    print("",file=f)

    print('Final results - MEDIAN',file=f)
    print("",file=f)

    print('Acc. without sampling:\t\t\t\t%.4f' % np.median(results_no_sampl),file=f)
    print('Acc. graphon sampling:\t\t\t%.4f' % (np.median(results_w_samp)),file=f)
    print('Acc. random sampling:\t\t\t\t%.4f' % (np.median(results_random_samp)),file=f)
    print("",file=f)

# Pickling
dict_results = {'results_no_sampl': results_no_sampl,
                'results_w_samp': results_w_samp,
                'results_random_samp': results_random_samp,
                'n_iters': n_iters_per_rlz}
pkl.dump(dict_results, open(os.path.join(saveDir,'results.p'), "wb"))
