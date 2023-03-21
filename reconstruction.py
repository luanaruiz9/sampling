# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:26:34 2023

@author: Luana Ruiz
"""

import numpy as np
import scipy.optimize as opt
import scipy
import torch

tol=1e-4

def f_rec(x, *args):
    L, m = args
    n = L.shape[0]
    y = torch.tensor(x,device=L.device,dtype=torch.float32)
    for i in range(m):
        y = torch.matmul(L,y)
        
    y = y.cpu().numpy()
    
    return np.linalg.norm(y)

def reconstruct(f, x, padded_eig, L, m):
    mask = padded_eig.cpu().numpy() > 0
    constr_var = np.diag(mask)
    constr_val = padded_eig
    constraint = opt.LinearConstraint(constr_var, lb=constr_val-tol, ub=constr_val+tol)
    
    res = opt.minimize(f, x, args=(L, m), method='SLSQP',
                       constraints=(constraint,), options={'disp': True})
    return res.x