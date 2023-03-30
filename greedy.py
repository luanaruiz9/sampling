# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 09:47:37 2023

@author: Luana Ruiz
"""

import numpy as np
import scipy.optimize as opt
import scipy
import torch

def f(x, *args):
    
    L = L.to('cpu')
    lam, L, k, s_vec = args
    n = L.shape[0]
    # Construct I_vsc from s_vec
    Ivsc = torch.zeros((n,n-int(np.sum(s_vec)))).to('cpu')
    # Construct I_scv from s_vec
    Iscv = torch.zeros((n-int(np.sum(s_vec)),n)).to('cpu')
    idx = np.argwhere(s_vec==0)
    for i in range(n-int(np.sum(s_vec))):
        Ivsc[idx[i],i] = 1
        Iscv[i,idx[i]] = 1
        
    omega = torch.matmul(Ivsc,torch.tensor(x,dtype=torch.float32)).to('cpu')
    for i in range(k):
        omega = torch.matmul(L,omega)
    Lt = torch.t(L)
    for i in range(k):
        omega = torch.matmul(Lt,omega)
    omega = torch.matmul(Iscv,omega)
    omega = torch.matmul(torch.tensor(x,dtype=torch.float32),omega)
     
    omega = omega.numpy()
    omega = np.power(np.linalg.norm(omega)/np.power(np.linalg.norm(x),2),1/2*k)
    
    return lam-omega

def f_lobpcg(x, *args):
    
    lam, L, k, s_vec = args
    n = L.shape[0]
    # Construct I_vsc from s_vec
    Ivsc = torch.zeros((n,n-int(np.sum(s_vec))))
    # Construct I_scv from s_vec
    Iscv = torch.zeros((n-int(np.sum(s_vec)),n))
    idx = np.argwhere(s_vec==0)
    for i in range(n-int(np.sum(s_vec))):
        Ivsc[idx[i],i] = 1
        Iscv[i,idx[i]] = 1
        
    omega = Ivsc
    for i in range(k):
        omega = torch.matmul(L,omega)
    Lt = torch.t(L)
    for i in range(k):
        omega = torch.matmul(Lt,omega)
    omega = torch.matmul(Iscv,omega)
     
    #omega = omega.cpu().numpy()
    x = torch.tensor(x,device=L.device,dtype=torch.float32)
    if len(x.shape) < 2:
        x = torch.unsqueeze(x,axis=1)
    omega, x = torch.lobpcg(A=omega,X=x,largest=False)
    return lam-np.power(omega,1/2*k)

def greedy(f, lam, L, k, m, exponent=5): # m is sampling set size
    
    n = L.shape[0]
    s_vec = np.zeros(n)
    idx_x = np.arange(n)
    for i in range(m):
        print(i)
        if np.random.rand(1) <= np.exp(-exponent*i/m):
            amax = idx_x[np.random.choice(n-i)]
            while s_vec[amax] == 1:
                amax = idx_x[np.random.choice(n-i)]
        else:
            x0 = np.random.multivariate_normal(np.zeros(n-i),np.eye(n-i)/np.sqrt(n-i))
            #x0 = np.ones(n-i)/(n-i)
            #x0 = np.zeros(n-i)
            #x0[np.random.choice(n-i)]=1
            res = opt.minimize(f, x0, args=(lam, L.to('cpu'), k, s_vec),method='CG',
                               options={'disp': True})
            #res = opt.minimize(f_lobpcg, np.expand_dims(x0,axis=1), args=(lam, L, k, s_vec),
            #                   options={'disp': True,'maxiter' : 10})
            phi = np.power(res.x,2)
            amax = idx_x[np.argmax(phi)]
        s_vec[amax] = 1
        idx_x = idx_x[s_vec[idx_x]==0]
    return s_vec
        