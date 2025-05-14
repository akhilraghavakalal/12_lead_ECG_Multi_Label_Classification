from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import os
from abc import ABC, abstractmethod


class BaseTrainer:
    def __init__(self, model, optimizer, criterion, device='cuda', 
                 model_dir='models', experiment_name='experiment'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_dir = os.path.join(model_dir, experiment_name)
        os.makedirs(self.model_dir, exist_ok=True)
        
    @abstractmethod
    def train_epoch(self, train_loader):
        pass
    
    @abstractmethod
    def validate(self, val_loader):
        pass
    
    def save_checkpoint(self, epoch, val_metric, filename='checkpoint.pth'):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metric': val_metric,
        }, os.path.join(self.model_dir, filename))
    
    def load_checkpoint(self, filename='checkpoint.pth'):
        checkpoint = torch.load(os.path.join(self.model_dir, filename))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['val_metric']

class ECGTrainer(BaseTrainer):
    def __init__(self, model, optimizer, device='cuda', 
                 model_dir='models', experiment_name='experiment'):
        super().__init__(model, optimizer, nn.BCELoss(), device, model_dir, experiment_name)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc='Training')
        
        for batch_idx, (ecg, wide_features, labels) in enumerate(progress_bar):
            ecg = ecg.to(self.device)
            wide_features = wide_features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(ecg, wide_features)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (batch_idx + 1)})
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for ecg, wide_features, labels in tqdm(val_loader, desc='Validating'):
                ecg = ecg.to(self.device)
                wide_features = wide_features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(ecg, wide_features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        aucs = []
        for i in range(all_outputs.shape[1]):
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
                aucs.append(auc)
        
        return total_loss / len(val_loader), np.mean(aucs)

def train_model(trainer, train_loader, val_loader, num_epochs=30, patience=5):
    best_val_metric = 0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_metric = trainer.validate(val_loader)
        
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Metric: {val_metric:.4f}')
        
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            patience_counter = 0
            trainer.save_checkpoint(epoch, val_metric, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after epoch {epoch+1}')
                break
        
        trainer.save_checkpoint(epoch, val_metric, 'last_model.pth')
    
    return best_val_metric