# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:26:34 2023

@author: Luana Ruiz
"""

import numpy as np
import scipy.optimize as opt
import torch

tol=1e-4

def hess(x,*args):
    n = x.shape[0]
    return np.zeros((n,n))

def f_rec(x, *args):
    L, m = args
    y = torch.tensor(x,device=L.device,dtype=torch.float32)
    for i in range(m):
        y = torch.matmul(L,y)
    y = y.cpu().numpy()
    return np.linalg.norm(y)

def reconstruct(f, x, padded_eig_lb, padded_eig_ub, L, m):
    n = x.shape[0]
    constr_var = np.eye(n)
    constr_val_lb = padded_eig_lb.cpu().numpy()
    constr_val_ub = padded_eig_ub.cpu().numpy()
    constraint = opt.LinearConstraint(constr_var, lb=constr_val_lb, ub=constr_val_ub)
    
    res = opt.minimize(f, x, args=(L, m), method='trust-constr',
                       constraints=(constraint,), tol=tol, hess=hess, 
                       options={'disp': True})
    return res.x