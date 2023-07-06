# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:02:23 2023

@author: Luana Ruiz
"""

import torch
import time
from tqdm import trange
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from torch_geometric.loader import DataLoader
import copy
import numpy as np

def train(model, train_data, val_data, optimizer, criterion, batch_size, n_epochs):

    loader = DataLoader([train_data], batch_size=batch_size)  

    # train
    losses = []
    val_accs = []
    best_acc = 0
    best_model = copy.deepcopy(model)
    start = time.time()
    for epoch in trange(n_epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.y
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        losses.append(total_loss)

        if epoch % 1 == 0:
          val_acc = test(model, val_data, is_validation=True)
          val_accs.append(val_acc)
          if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)
        else:
          val_accs.append(val_accs[-1])
        end = time.time()
    final_model = model
    training_time = end-start
    return val_accs, losses, best_model, final_model, best_acc, loader, training_time

def test(test_model, data, is_validation=False, save_model_preds=False):
    
    loader = DataLoader([data])
    test_model.eval()
    correct = 0
    # Note that Cora is only one graph!
    for data in loader:
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = test_model(data).max(dim=1)[1]
            label = data.y

        mask = data.val_mask if is_validation else data.test_mask
        # node classification: only evaluate on nodes in test set
        pred = pred[mask]
        label = label[mask]

        if save_model_preds:
          print ("Saving Model Predictions for Model Type", test_model.type)

          data = {}
          data['pred'] = pred.view(-1).cpu().detach().numpy()
          data['label'] = label.view(-1).cpu().detach().numpy()
            
        correct += pred.eq(label).sum().item()

    total = 0
    for data in loader.dataset:
        total += torch.sum(data.val_mask if is_validation else data.test_mask).item()

    return correct / total

def train_link_predictor(model, train_data, val_data, optimizer, criterion, n_epochs=100):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
    
    best_val_auc = 0
    best_model = None
    for epoch in range(1, n_epochs + 1):

        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index, train_data.eigvecs)

        # sampling training negatives for every training epoch
        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1), method='sparse')

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()

        val_auc = eval_link_predictor(model, val_data)
        if val_auc >= best_val_auc:
            best_model = copy.deepcopy(model)
            best_val_auc = val_auc

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.3f}, Val AUC: {val_auc:.3f}")
            
        scheduler.step()
        
    if best_model is None:
        best_model = model
    return best_model


@torch.no_grad()
def eval_link_predictor(model, data):

    model.eval()
    z = model.encode(data.x, data.edge_index, data.eigvecs)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()

    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())