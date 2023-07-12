#!/bin/bash
python main_link_pred.py cora 0.0013233812694319355 100 10 20 10 10 2 32 64
python main_link_pred_wandb.py citeseer 0.004377884943245985 100 10 20 10 20 4 128 32
python main_link_pred_wandb.py pubmed 0.00293072250746189 100 50 30 20 2 32 128
python main_link_pred_wandb.py twitch-pt 0.00481447754363471 100 20 10 20 2 32 128
python main_link_pred_wandb.py twitch-ru 0.0024717797601309043 100 20 10 10 2 64 64