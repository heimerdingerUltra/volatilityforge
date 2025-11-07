import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np


class AttentionEnsemble(nn.Module):
    
    def __init__(self, n_models: int, hidden_dim: int = 128):
        super().__init__()
        
        self.n_models = n_models
        
        self.query = nn.Linear(1, hidden_dim)
        self.key = nn.Linear(n_models, hidden_dim)
        self.value = nn.Linear(n_models, n_models)
        
        self.output = nn.Sequential(
            nn.Linear(n_models, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, predictions: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = predictions.shape[0]
        
        if target is not None:
            q = self.query(target.unsqueeze(-1))
        else:
            mean_pred = predictions.mean(dim=1, keepdim=True)
            q = self.query(mean_pred)
        
        k = self.key(predictions)
        v = self.value(predictions)
        
        attention = torch.matmul(q, k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
        attention = F.softmax(attention, dim=-1)
        
        weighted = torch.matmul(attention, v)
        
        return self.output(weighted).squeeze(-1)


class HierarchicalEnsemble(nn.Module):
    
    def __init__(self, n_models: int, n_groups: int = 2):
        super().__init__()
        
        self.n_models = n_models
        self.n_groups = n_groups
        
        models_per_group = n_models // n_groups
        
        self.group_combiners = nn.ModuleList([
            nn.Sequential(
                nn.Linear(models_per_group, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )
            for _ in range(n_groups)
        ])
        
        self.meta_combiner = nn.Sequential(
            nn.Linear(n_groups, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )
        
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        models_per_group = self.n_models // self.n_groups
        
        group_outputs = []
        for i, combiner in enumerate(self.group_combiners):
            start_idx = i * models_per_group
            end_idx = start_idx + models_per_group
            group_preds = predictions[:, start_idx:end_idx]
            group_out = combiner(group_preds)
            group_outputs.append(group_out)
        
        group_outputs = torch.cat(group_outputs, dim=1)
        
        return self.meta_combiner(group_outputs).squeeze(-1)


class UncertaintyEnsemble(nn.Module):
    
    def __init__(self, n_models: int, hidden_dim: int = 64):
        super().__init__()
        
        self.mean_head = nn.Sequential(
            nn.Linear(n_models, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Linear(n_models, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
    def forward(self, predictions: torch.Tensor) -> tuple:
        mean = self.mean_head(predictions).squeeze(-1)
        uncertainty = self.uncertainty_head(predictions).squeeze(-1)
        return mean, uncertainty


class AdvancedEnsembleSystem:
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        device: str = 'cuda',
        ensemble_type: str = 'attention'
    ):
        self.models = {name: model.to(device) for name, model in models.items()}
        self.device = device
        self.ensemble_type = ensemble_type
        
        n_models = len(models)
        
        if ensemble_type == 'attention':
            self.combiner = AttentionEnsemble(n_models).to(device)
        elif ensemble_type == 'hierarchical':
            self.combiner = HierarchicalEnsemble(n_models).to(device)
        elif ensemble_type == 'uncertainty':
            self.combiner = UncertaintyEnsemble(n_models).to(device)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        
    def get_base_predictions(self, X: torch.Tensor) -> torch.Tensor:
        predictions = []
        
        for model in self.models.values():
            model.eval()
            with torch.no_grad():
                pred = model(X)
                predictions.append(pred)
        
        return torch.stack(predictions, dim=1)
    
    def forward(self, X: torch.Tensor, return_uncertainty: bool = False) -> torch.Tensor:
        base_preds = self.get_base_predictions(X)
        
        if self.ensemble_type == 'uncertainty' and return_uncertainty:
            return self.combiner(base_preds)
        else:
            return self.combiner(base_preds)
    
    def train_combiner(
        self,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        n_epochs: int = 50,
        patience: int = 10,
        verbose: bool = True
    ):
        
        for model in self.models.values():
            model.eval()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            self.combiner.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                base_preds = self.get_base_predictions(X_batch)
                
                optimizer.zero_grad()
                
                if self.ensemble_type == 'uncertainty':
                    pred, uncertainty = self.combiner(base_preds)
                    nll_loss = 0.5 * ((pred - y_batch) ** 2) / (uncertainty + 1e-6) + 0.5 * torch.log(uncertainty + 1e-6)
                    loss = nll_loss.mean()
                else:
                    pred = self.combiner(base_preds)
                    loss = criterion(pred, y_batch)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            self.combiner.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    base_preds = self.get_base_predictions(X_batch)
                    
                    if self.ensemble_type == 'uncertainty':
                        pred, _ = self.combiner(base_preds)
                    else:
                        pred = self.combiner(base_preds)
                    
                    loss = criterion(pred, y_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} - train: {train_loss:.4f} - val: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.combiner.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
        
        self.combiner.load_state_dict(best_state)
    
    def predict(self, X: torch.Tensor, return_uncertainty: bool = False) -> np.ndarray:
        X = X.to(self.device)
        self.combiner.eval()
        
        with torch.no_grad():
            if self.ensemble_type == 'uncertainty' and return_uncertainty:
                pred, uncertainty = self.forward(X, return_uncertainty=True)
                return pred.cpu().numpy(), uncertainty.cpu().numpy()
            else:
                pred = self.forward(X)
                return pred.cpu().numpy()
    
    def predict_with_diversity(self, X: torch.Tensor) -> Dict[str, np.ndarray]:
        X = X.to(self.device)
        
        base_preds = self.get_base_predictions(X)
        base_preds_np = base_preds.cpu().numpy()
        
        ensemble_pred = self.forward(X).cpu().numpy()
        
        return {
            'prediction': ensemble_pred,
            'mean': base_preds_np.mean(axis=1),
            'std': base_preds_np.std(axis=1),
            'min': base_preds_np.min(axis=1),
            'max': base_preds_np.max(axis=1),
            'individual': base_preds_np
        }
