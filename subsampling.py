# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:14:26 2023

@author: Luana Ruiz
"""

import torch
import numpy as np

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

def cluster_hk_pr(A, seed, cheeger, eps, sz, vol):

    n = A.shape[0]
    A[0:n,0:n] = torch.ones(n)
    vol_G = torch.sum(A).cpu().numpy()
    
    if sz is None:
        sz = int(n/2)
    if vol is None:
        vol = np.floor(vol_G)/4
        
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
            return []
        elif volS >= vol/2 and volS <= 2*vol and cheegerS <= np.sqrt(8*cheeger):
            return S
        
def sample_clustering(A, m, nb_cuts=1, cheeger=0.5, eps=0.05, sz=None, vol=None):  
    
    m_per_cluster = np.floor(m/(nb_cuts+1))
    cluster_sizes = []
    for i in range(nb_cuts-1):
        cluster_sizes.append(int(m_per_cluster))
    cur_sum  = np.sum(np.array(cluster_sizes))
    cluster_sizes.append(int(m-cur_sum))
    clusters = []
    S_complement = list(np.arange(A.shape[0]))
    for i in range(nb_cuts):
        thisA = A[S_complement,:]
        thisA = thisA[:,S_complement]
        thisSeed = np.random.choice(thisA.shape[0])
        idx = cluster_hk_pr(thisA, thisSeed, cheeger, eps, sz, vol)
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
    
        
    
            