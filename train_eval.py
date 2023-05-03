# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 14:43:24 2023

@author: Luana Ruiz
"""

import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
import copy
from torch_geometric.loader import LinkNeighborLoader

def train_link_predictor(model, train_data, val_data, optimizer, criterion, n_epochs=100, batch_size=None):
    if batch_size == None:
        batch_size = train_data.edge_index.shape[1]
    train_loader = LinkNeighborLoader(train_data, num_neighbors=[-1] * 2, 
                                          edge_label_index=train_data.edge_label_index,
                                          edge_label=train_data.edge_label,
                                          batch_size=batch_size, shuffle=True)
    
    best_val_auc = 0
    best_model = None
    for epoch in range(1, n_epochs + 1):
        total_loss = 0
        model.train()
        for i, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            z = model.encode(batch_data.x, batch_data.edge_index, batch_data.eigvecs)
    
            # sampling training negatives for every training epoch
            neg_edge_index = negative_sampling(
                edge_index=batch_data.edge_index, num_nodes=batch_data.num_nodes,
                num_neg_samples=batch_data.edge_label_index.size(1), method='sparse')
    
            edge_label_index = torch.cat(
                [batch_data.edge_label_index, neg_edge_index],
                dim=-1,
            )
            edge_label = torch.cat([
                batch_data.edge_label,
                batch_data.edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0)
    
            out = model.decode(z, edge_label_index).view(-1)
            loss = criterion(out, edge_label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            if (epoch) % 10 == 0:
                val_auc = eval_link_predictor(model, val_data)
                if val_auc > best_val_auc:
                    best_model = copy.deepcopy(model)
                    best_val_auc = val_auc
                total_loss /= (i + 1)
                print(f"Epoch: {epoch:03d}, Train Loss: {total_loss:.3f}, Val AUC: {val_auc:.3f}")
            

    
        #if epoch % 10 == 0:
        #    print(f"Epoch: {epoch:03d}, Train Loss: {total_loss:.3f}, Val AUC: {val_auc:.3f}")
    if best_model is None:
        best_model = model
        
    del train_loader
    
    return best_model


@torch.no_grad()
def eval_link_predictor(model, data):

    model.eval()
    z = model.encode(data.x, data.edge_index, data.eigvecs)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()

    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())