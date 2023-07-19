# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 09:47:37 2023

@author: Luana Ruiz
"""

import numpy as np
import scipy.optimize as opt
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, is_undirected

###########################
# Generate induced graphon

# Sampling will be performed on the graphon (induced by the large graph)


def generate_induced_graphon(og_graph, n_intervals):
    # Here, we are actually generating the graph associated with the induced graphon.
    # This explains the normalizations used here.
    n = og_graph.num_nodes
    
    n_nodes_per_int, n_nodes_last_int = np.divmod(n,n_intervals)
    if n_nodes_last_int == 0:
        n_nodes_last_int = n_nodes_per_int
    edge_index = og_graph.edge_index
    
    if og_graph.edge_weight is not None:
        edge_weight = og_graph.edge_weight
    else:
        edge_weight = torch.ones(edge_index.shape[1],device=edge_index.device)
        
    adj_sparse = torch.sparse_coo_tensor(edge_index, edge_weight, (n, n))
    adj = adj_sparse.to_dense()
    adj_ind_graphon = torch.zeros(n_intervals,n_intervals,device=edge_index.device)
    x = og_graph.x
    x_ind_graphon = torch.zeros(n_intervals,og_graph.x.shape[-1],device=edge_index.device)
    y = og_graph.y
    y_ind_graphon = torch.zeros(n_intervals,device=edge_index.device)
    
    for i in range(n_intervals):
        for j in range(n_intervals):
            if i < n_intervals-1 and j < n_intervals-1:
                adj_ind_graphon[i,j] = torch.sum(adj[i*n_nodes_per_int:(i+1)*n_nodes_per_int,
                                             j*n_nodes_per_int:(j+1)*n_nodes_per_int])/n_nodes_per_int
            elif i < n_intervals-1 and j == n_intervals-1:
                adj_ind_graphon[i,j] = torch.sum(adj[i*n_nodes_per_int:(i+1)*n_nodes_per_int,
                                             j*n_nodes_per_int:-1])/np.sqrt(n_nodes_per_int*n_nodes_last_int)
            elif j < n_intervals-1 and i == n_intervals-1:
                adj_ind_graphon[i,j] = torch.sum(adj[i*n_nodes_per_int:-1,
                                             j*n_nodes_per_int:(j+1)*n_nodes_per_int])/np.sqrt(n_nodes_per_int*n_nodes_last_int)
            else:
                adj_ind_graphon[i,j] = torch.sum(adj[i*n_nodes_per_int:-1,
                                             j*n_nodes_per_int:-1])/n_nodes_last_int
        # The signals below are not used, for now. Note that y is categorical.
        if i < n_intervals-1:
            x_ind_graphon[i] = torch.sum(x[i*n_nodes_per_int:(i+1)*n_nodes_per_int],axis=0)/np.sqrt(n_nodes_per_int)
            y_ind_graphon[i] = torch.sum(y[i*n_nodes_per_int:(i+1)*n_nodes_per_int])/n_nodes_per_int > 0.5
        else:
            x_ind_graphon[i] = torch.sum(x[i*n_nodes_per_int:-1],axis=0)/np.sqrt(n_nodes_last_int)
            y_ind_graphon[i] = torch.sum(y[i*n_nodes_per_int:-1])/n_nodes_last_int > 0.5
    edge_index_ind_graphon, edge_weight_ind_graphon = dense_to_sparse(adj_ind_graphon)
    assert is_undirected(edge_index_ind_graphon)

    data = Data(x=x_ind_graphon,edge_index=edge_index_ind_graphon,
                edge_weight=edge_weight_ind_graphon,y=y_ind_graphon)
    
    return data

############################
# Greedy sampling algorithm

def f(x, *args):
    lam, L, k, s_vec = args
    n = L.shape[0]
    # Construct I_vsc from s_vec
    Ivsc = np.zeros((n,n-int(np.sum(s_vec))))
    idx = np.argwhere(s_vec==0).squeeze()
    for i in range(n-int(np.sum(s_vec))):
        Ivsc[idx[i],i] = 1
        
    omega = np.matmul(Ivsc,x)
    omega = torch.tensor(omega,dtype=torch.float32)
    for i in range(k):
        omega = torch.matmul(L,omega)
    omega = omega.numpy()
    omega = np.power(np.dot(omega,omega)/np.dot(x,x),1/(2*k))

    return omega

def greedy(f, lam, L, k, m, exponent=10000000): # m is sampling set size
    lam = lam.cpu().numpy()
    L = L.cpu()
    n = L.shape[0]
    s_vec = np.zeros(n)
    idx_x = np.arange(n)
    n_iters = 0
    print(L)
    for i in range(m):
        print(i)
        if False:#np.random.rand(1) <= np.exp(-exponent*i/m):
            amax = idx_x[np.random.choice(n-i)]
            while s_vec[amax] == 1:
                amax = idx_x[np.random.choice(n-i)]
        else:
            #x0 = np.random.multivariate_normal(np.zeros(n-i),np.eye(n-i))
            x0 = np.random.multivariate_normal(np.zeros(n),np.eye(n))
            x0 = torch.matmul(L,torch.tensor(x0).float()).cpu().numpy()
            x0 = x0[np.argwhere(s_vec==0).squeeze()]
            #x0 = np.ones(n-i)/(n-i)
            #x0 = np.zeros(n-i)
            #x0[np.random.choice(n-i)] += 1
            print(f(x0, lam, L, k, s_vec))
            print(x0)
            phi0 = np.power(x0,2)
            print(np.argmax(phi0))
            res = opt.minimize(f, x0, args=(lam, L, k, s_vec),
                                       options={'disp': True})
            n_iters += res.nit
            print(res.fun)
            print(res.x)
            print(res.x-x0)
            #res = opt.minimize(f_lobpcg, np.expand_dims(x0,axis=1), args=(lam, L, k, s_vec),
            #                   options={'disp': True,'maxiter' : 10})
            phi = np.power(res.x,2)
            amax = idx_x[np.argmax(phi)]
            print(amax)
        s_vec[amax] = 1
        idx_x = idx_x[s_vec[idx_x]==0]
        if res.success and res.fun > lam:
            print('here')
            print(res.fun)
            print(lam)
            break
    return s_vec, n_iters

##################################
# Local clustering-based sampling

# Heat kernel page rank, adapted from Chung et al.

def local_hk_pr(A, t, seed, eps):
    
    n = A.shape[0]
    rho = torch.zeros(n)
    
    r = int((16/(np.power(eps,1/3)))*np.log(n))
    c = 1
    K = c*np.log(1/eps)/(np.log(np.log(1/eps)))
    
    for i in range(r):
        k = K + 1
        while k > K:
            k = np.random.poisson(t)
        Ak = A 
        for j in range(k-1):
            Ak = torch.matmul(A, Ak)
        prob_vec = Ak[seed].cpu().numpy()
        last_node = np.random.choice(np.arange(n),p=prob_vec)
        rho[last_node] += 1
        
    return rho/r

def cluster_hk_pr(A, seed, cheeger, eps, sz, vol_den):

    n = A.shape[0]
    A[0:n,0:n] = torch.ones(n)
    vol_G = torch.sum(A).cpu().numpy()
    
    if sz is None:
        sz = int(n/2)
    if vol_den is None:
        vol = np.floor(vol_G/4)
    else:
        vol = np.floor(vol_G/vol_den)
        
    t = (1/cheeger)*np.log(2*np.sqrt(vol)/(1-eps) + 2*eps*sz)
    
    deg_vec = torch.sum(A, axis=1) 
    deg_inv = torch.diag(torch.pow(deg_vec,-1))
    #torch.nan_to_num(deg_inv)
    
    rho = local_hk_pr(torch.matmul(deg_inv,A), t, seed, eps)
    
    deg_vec= deg_vec.cpu()
    idx = torch.argsort(rho/deg_vec).numpy()
    
    S = []
    for i, node in enumerate(list(idx)):
        S.append(node)
        volS = torch.sum(deg_vec[S]).cpu().numpy()
        bS = list(torch.argwhere(A[S])[1].cpu().numpy())
        bS = list(set(bS) - set(S))
        denominator = min(volS,vol_G-volS)
        cheegerS = len(bS)/denominator
        if volS > 2*vol:
            return list(np.random.choice(np.arange(n),sz)) # return a random 
                                                           # set of size sz if 
                                                           # nodes cannot be clustered
        elif volS >= vol/2 and volS <= 2*vol and cheegerS <= np.sqrt(8*cheeger):
            return S
        
def sample_clustering(A, m, nb_cuts=1, cheeger=0.5, eps=0.05, sz=None, vol_den=4):  
    if nb_cuts > 3:
        vol_den = nb_cuts + 1
    m_per_cluster = np.floor(m/(nb_cuts+1))
    cluster_sizes = []
    for i in range(nb_cuts):
        cluster_sizes.append(int(m_per_cluster))
    cur_sum  = np.sum(np.array(cluster_sizes))
    cluster_sizes.append(int(m-cur_sum))
    clusters = []
    S_complement = list(np.arange(A.shape[0]))
    for i in range(nb_cuts):
        thisA = A[S_complement,:]
        thisA = thisA[:,S_complement]
        if thisA.shape[0] == 0:
            break
        thisSeed = np.random.choice(thisA.shape[0])
        idx = cluster_hk_pr(thisA, thisSeed, cheeger, eps, cluster_sizes[i],vol_den)
        S = []
        for j in idx:
            S.append(S_complement[j])
        clusters.append(S)
        for elt in S:
            S_complement.remove(elt)
    clusters.append(S_complement)
    
    sampled_idx = []
    for cluster, sz in zip(clusters,cluster_sizes):
        cluster_arr = np.array(cluster)
        if sz > len(cluster_arr):
            sampled_idx += list(cluster_arr)
        else:
            sampled_idx += list(np.random.choice(cluster_arr,sz,replace=False))
    
    return sampled_idx

###

'''
# !!!!!!
# Deprecated, fix if speedups are needed!
def f_lobpcg(x, *args):
    lam, L, k, s_vec = args
    n = L.shape[0]
    # Construct I_vsc from s_vec
    Ivsc = np.zeros((n,n-int(np.sum(s_vec))))
    # Construct I_scv from s_vec
    Iscv = torch.zeros((n-int(np.sum(s_vec)),n))
    idx = np.argwhere(s_vec==0)
    for i in range(n-int(np.sum(s_vec))):
        Ivsc[idx[i],i] = 1
        Iscv[i,idx[i]] = 1
        
    omega = Ivsc
    omega = torch.tensor(omega,dtype=torch.float32)
    for i in range(k):
        omega = torch.matmul(L,omega)
    Lt = torch.t(L)
    for i in range(k):
        omega = torch.matmul(Lt,omega)
    omega = torch.matmul(Iscv,omega)
    #omega = omega.cpu().numpy()
    x = torch.tensor(x,dtype=torch.float32)
    if len(x.shape) < 2:
        x = torch.unsqueeze(x,axis=1)
    omega, x = torch.lobpcg(A=omega,X=x,largest=False)
    return lam-np.power(omega.numpy(),1/2*k)    
'''  