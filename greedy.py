# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 09:47:37 2023

@author: Luana Ruiz
"""

import numpy as np
import scipy.optimize as opt
import torch

def f(x, *args):
    
    lam, L, k, s_vec, alpha = args
    omega = torch.tensor(x,dtype=torch.float32)
    for i in range(k):
        omega = torch.matmul(L,omega)
    omega = omega.cpu().numpy()
    omega = np.power(np.linalg.norm(omega)/np.linalg.norm(x),k)
    
    diag1s = np.zeros(L.shape)
    np.fill_diagonal(diag1s,s_vec)
    diag_term = np.dot(x,np.matmul(diag1s,x))
    diag_term = diag_term/np.linalg.norm(x)
    
    return np.abs(lam-omega) + alpha*diag_term

def greedy(f, x0, lam, L, k, m): # m is sampling set size
    
    alpha = 10
    n = L.shape[0]
    s_vec = np.zeros(n)
    for i in range(m):
        print(i)
        res = opt.minimize(f, x0, args=(lam, L, k, s_vec, alpha),
                           options={'disp': True,'maxiter' : 10})
        phi = np.power(res.x,2)
        s_vec[np.argmax(phi)] = 1
        
    return s_vec
        