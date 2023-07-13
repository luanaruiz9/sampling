#!/bin/bash
python main_link_pred.py cora 0.001 100 10 50 25 30 1 32 128
python main_link_pred.py citeseer 0.001 100 10 50 25 30 1 32 32
python main_link_pred.py pubmed 0.003 100 10 50 25 60 2 32 128
python main_link_pred.py twitch-pt 0.001 100 10 50 25 30 1 32 32
python main_link_pred.py twitch-ru 0.003 100 10 50 25 30 1 32 128