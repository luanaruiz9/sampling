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
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Twitch
from ogb.linkproppred import PygLinkPropPredDataset

import torch_geometric.transforms as T
from torch_geometric.data import Data

from architecture import  SignNetLinkPredNet
from train_eval_ogb import train_link_predictor, eval_link_predictor
from greedy import greedy, f
#from reconstruction import f_rec, reconstruct
from subsampling import sample_clustering
from graphon_sampling import generate_induced_graphon
import aux_functions

data_name = sys.argv[1]
lr = float(sys.argv[2])
n_epochs = int(sys.argv[3])
ratio_train = 0#float(sys.argv[4])
ratio_test = 0#float(sys.argv[5])
ratio_val = 1-ratio_train-ratio_test
n_realizations = 1 #10
m = int(sys.argv[3]) #50 # Number of candidate intervals
m2 = int(sys.argv[5]) #25 # Number of sampled intervals
m3 = int(sys.argv[6]) #3 #8 # How many nodes (points) to sample per sampled interval
nb_cuts = int(sys.argv[7])

F_nn = [128, 128]
F_pe = [64, 64]


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
    
K = 20
do_no_pe = True
do_eig = True
do_learn_pe = True
do_w_sampl = True
do_random_sampl = True

"""
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
"""

dataset = PygLinkPropPredDataset(name="ogbl-ddi") 
    
graph_og = dataset[0]
graph_og = graph_og.subgraph(torch.arange(500)) # comment it out
pre_defined_kwargs = {'eigvecs': False}
graph = Data(x=graph_og.x, edge_index=graph_og.edge_index, 
             edge_weight=graph_og.edge_weight, y=graph_og.y,**pre_defined_kwargs)
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

for r in range(n_realizations):
    
    print('Realization ' + str(r))
    print()
    
    split_edge = dataset.get_edge_split()
    train_data, val_data, test_data = split_edge["train"], split_edge["valid"], split_edge["test"]
    #graph = dataset[0] # pyg graph object containing only training edges
    
    if do_no_pe:
    
        model = SignNetLinkPredNet(dataset.num_features, F_nn[0], F_nn[1]).to(device)
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
    eigvals, V = torch.lobpcg(L, k=K)
    
    # V for test data
    adj_sparse_test, adj_test = aux_functions.compute_adj_from_data(test_data)
    
    # Computing normalized Laplacian
    L_test = aux_functions.compute_laplacian(adj_sparse_test, num_nodes)
    eigvals_test, V_test = torch.lobpcg(L_test, k=K)
    
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
        
        model = SignNetLinkPredNet(dataset.num_features+K, F_nn[0], F_nn[1]).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        model, _, _ = train_link_predictor(model, train_data_new, val_data_new, optimizer, 
                                     criterion, n_epochs=n_epochs, K=K, ten_fold=False)
        
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
        
        model = SignNetLinkPredNet(dataset.num_features+F_pe[-1]*K, F_nn[0], F_nn[1], True, 1, F_pe[0], F_pe[1]).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        model, _, _ = train_link_predictor(model, train_data_new, val_data_new, optimizer, 
                                     criterion, n_epochs=n_epochs, K=K, pe=True, ten_fold=False)
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
        
        idx = torch.argsort(torch.abs(eigvals_test))
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
                    cur_adj = adj[i*n_nodes_per_int:(i+1)*n_nodes_per_int,
                                      i*n_nodes_per_int:(i+1)*n_nodes_per_int]
                    idx = sample_clustering(cur_adj, m3, nb_cuts=nb_cuts)#np.random.choice(np.arange(i*n_nodes_per_int,(i+1)*n_nodes_per_int), m3, replace=False)
                else:
                    if m3 > n_nodes_last_int:
                        #m3 = n_nodes_last_int
                        cur_adj = adj[i*n_nodes_per_int:i*n_nodes_per_int+n_nodes_last_int,
                                                i*n_nodes_per_int:i*n_nodes_per_int+n_nodes_last_int]
                    else:
                        cur_adj = adj[i*n_nodes_per_int:i*n_nodes_per_int+n_nodes_last_int,
                                                i*n_nodes_per_int:i*n_nodes_per_int+m3]
                    idx = sample_clustering(cur_adj, n_nodes_last_int, nb_cuts=nb_cuts)#np.random.choice(np.arange(i*n_nodes_per_int,
                                                     #i*n_nodes_per_int+n_nodes_last_int), m3, replace=False)
                idx = np.sort(idx)
                sampled_idx += list(idx)
        
        # V for train data
        graph_new = train_data.subgraph(torch.tensor(sampled_idx, device=device, dtype=torch.long))
        graph_new = graph_new.to(device)
        num_nodes_new = graph_new.x.shape[0]
        adj_sparse_new, adj_new = aux_functions.compute_adj_from_data(graph_new)
        
        # Computing normalized Laplacian
        L_new = aux_functions.compute_laplacian(adj_sparse_new, num_nodes_new)
        
        eigvals_new, V_new = torch.lobpcg(L_new, k=K)
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
        
        # V for test data
        graph_new = test_data.subgraph(torch.tensor(sampled_idx, device=device, dtype=torch.long))
        graph_new = graph_new.to(device)
        num_nodes_new = graph_new.x.shape[0]
        adj_sparse_new, adj_new = aux_functions.compute_adj_from_data(graph_new)
        
        # Computing normalized Laplacian
        L_new = aux_functions.compute_laplacian(adj_sparse_new, num_nodes_new)
        
        eigvals_new, V_new = torch.lobpcg(L_new, k=K)
        V_new = V_new.type(torch.float32)
        V_rec_test = torch.zeros(num_nodes, K, device=device)
        
        for i in range(V_new.shape[1]):
            v = V_new[:,i]
            #x0 = np.random.multivariate_normal(np.zeros(num_nodes), np.eye(num_nodes)/np.sqrt(num_nodes))
            #x0[sampled_idx] = v.cpu().numpy()*np.sqrt(m2*m3)/np.sqrt(num_nodes)
            #v_padded_lb = -torch.ones(num_nodes, device=device)
            #v_padded_lb[sampled_idx] = v*np.sqrt(m2*m3)/np.sqrt(num_nodes)
            #v_padded_ub = torch.ones(num_nodes, device=device)
            #v_padded_ub[sampled_idx] = v*np.sqrt(m2*m3)/np.sqrt(num_nodes)
            #V_rec_test[:,i] = torch.from_numpy(reconstruct(f_rec, x0, v_padded_lb, v_padded_ub, L, k))
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
        
        model = SignNetLinkPredNet(dataset.num_features+K, F_nn[0], F_nn[1]).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        model, train_data_collection, V_collection = train_link_predictor(model, train_data_new, val_data_new, optimizer, 
                                     criterion, n_epochs=n_epochs, K=K, m=m, m2=m2, 
                                     m3=m3, nb_cuts=nb_cuts, ten_fold=False)
        
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
        
        model = SignNetLinkPredNet(dataset.num_features+F_pe[-1]*K, F_nn[0], F_nn[1], True, 1, F_pe[0], F_pe[1]).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        model, _, _ = train_link_predictor(model, train_data_new, val_data_new, optimizer, 
                                     criterion, n_epochs=n_epochs, K=K, pe=True, m=m, 
                                     m2=m2, m3=m3, nb_cuts=nb_cuts, 
                                     train_data_collection=train_data_collection, 
                                     V_collection=V_collection, ten_fold=False)
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
        
        sampled_idx2 = list(np.random.choice(np.arange(num_nodes), m2*m3, replace=False))

        # V for train data
        graph_new = train_data.subgraph(torch.tensor(sampled_idx2, device=device, dtype=torch.long))
        graph_new = graph_new.to(device)
        num_nodes_new = graph_new.x.shape[0]
        adj_sparse_new, adj_new = aux_functions.compute_adj_from_data(graph_new)
        
        # Computing normalized Laplacian
        L_new = aux_functions.compute_laplacian(adj_sparse_new, num_nodes_new)
        
        eigvals_new, V_new = torch.lobpcg(L_new, k=K)
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
            
        # V for test data
        graph_new = test_data.subgraph(torch.tensor(sampled_idx2, device=device, dtype=torch.long))
        graph_new = graph_new.to(device)
        num_nodes_new = graph_new.x.shape[0]
        adj_sparse_new, adj_new = aux_functions.compute_adj_from_data(graph_new)
        
        # Computing normalized Laplacian
        L_new = aux_functions.compute_laplacian(adj_sparse_new, num_nodes_new)
        
        eigvals_new, V_new = torch.lobpcg(L_new, k=K)
        V_new = V_new.type(torch.float32)
        V_rec_test = torch.zeros(num_nodes, K, device=device)
        
        for i in range(V_new.shape[1]):
            v = V_new[:,i]
            #x0 = np.random.multivariate_normal(np.zeros(num_nodes), np.eye(num_nodes)/np.sqrt(num_nodes))
            #x0[sampled_idx2] = v.cpu().numpy()*np.sqrt(m2*m3)/np.sqrt(num_nodes)
            #v_padded_lb = -torch.ones(num_nodes, device=device)
            #v_padded_lb[sampled_idx2] = v*np.sqrt(m2*m3)/np.sqrt(num_nodes)
            #v_padded_ub = torch.ones(num_nodes, device=device)
            #v_padded_ub[sampled_idx2] = v*np.sqrt(m2*m3)/np.sqrt(num_nodes)
            #V_rec_test[:,i] = torch.from_numpy(reconstruct(f_rec, x0, v_padded_lb, v_padded_ub, L, k))
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
        
        model = SignNetLinkPredNet(dataset.num_features+K, F_nn[0], F_nn[1]).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        model, train_data_collection, V_collection = train_link_predictor(model, train_data_new, val_data_new, optimizer, 
                                     criterion, n_epochs=n_epochs, K=K, m2=m2, m3=m3, ten_fold=False)
        
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
        
        model = SignNetLinkPredNet(dataset.num_features+F_pe[-1]*K, F_nn[0], F_nn[1], True, 1, F_pe[0], F_pe[1]).to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        model, _, _ = train_link_predictor(model, train_data_new, val_data_new, optimizer, 
                                     criterion, n_epochs=n_epochs, K=K, pe=True, m2=m2, m3=m3,
                                     train_data_collection=train_data_collection,
                                     V_collection=V_collection, ten_fold=False)
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

    print('Avg. AUC w/o eigenvectors:\t\t\t\t\t%.4f' % np.mean(results_no_eigs),file=f)
    print('Avg. AUC w/ eigenvectors and w/ PEs:\t\t\t\t%.4f    %.4f' % (np.mean(results_eigs), np.mean(results_pe)),file=f)
    print('Avg. AUC graphon sampling, idem above:\t\t\t%.4f    %.4f' % (np.mean(results_w_samp_eigs), np.mean(results_w_samp_pe)),file=f)
    print('Avg. AUC random sampling, idem above:\t\t\t\t%.4f    %.4f' % (np.mean(results_random_samp_eigs), np.mean(results_random_samp_pe)),file=f)
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
