#!/bin/bash

python main_link_pred.py cora 0.001 100 10 20 10 20 1 32 32 20
python main_link_pred.py citeseer 0.001 100 10 20 10 20 1 32 32 20
python main_link_pred.py pubmed 0.001 100 10 50 25 30 1 32 32 20
python main_transf.py cora 0.001 100 10 20 10 10 1 32 7
python main_transf.py cora 0.001 100 10 20 10 20 1 32 7
python main_transf.py cora 0.001 100 10 30 15 20 1 32 7
python main_transf.py citeseer 0.001 100 10 20 10 10 1 32 6
python main_transf.py citeseer 0.001 100 10 20 10 20 1 32 6
python main_transf.py citeseer 0.001 100 10 30 15 20 1 32 6
python main_transf.py pubmed 0.001 100 10 20 10 50 1 32 3
python main_transf.py pubmed 0.001 100 10 20 10 20 1 32 3
python main_transf.py pubmed 0.001 100 10 30 15 20 1 32 3
