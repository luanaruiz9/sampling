# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:22:26 2023

@author: Luana Ruiz
"""

sweep_config = {
                'method': 'random',
                'metric': {'goal': 'minimize', 'name': 'loss'},
                'parameters': {
                    'n_epochs': {'value': 1000},
                    'F_nn': {'values': [32, 64, 128]},
                    'F_pe': {'values': [32, 64, 128]},
                    'lr': {'distribution': 'uniform',
                                      'max': 0.01,
                                      'min': 0}
                }
 }