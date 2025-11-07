import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import joblib


class OptionsDataset(Dataset):
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class DataPipeline:
    
    def __init__(
        self,
        batch_size: int = 512,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        
    def create_loaders(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.1,
        seed: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        
        n = len(X)
        indices = np.random.RandomState(seed).permutation(n)
        
        test_split = int(n * test_size)
        val_split = int(n * (test_size + val_size))
        
        test_idx = indices[:test_split]
        val_idx = indices[test_split:val_split]
        train_idx = indices[val_split:]
        
        dataset = OptionsDataset(X, y)
        
        train_loader = self._create_loader(
            Subset(dataset, train_idx),
            shuffle=True
        )
        
        val_loader = self._create_loader(
            Subset(dataset, val_idx),
            shuffle=False
        )
        
        test_loader = self._create_loader(
            Subset(dataset, test_idx),
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader
    
    def create_kfold_loaders(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        seed: int = 42
    ):
        y_binned = pd.cut(y, bins=10, labels=False)
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        dataset = OptionsDataset(X, y)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
            train_loader = self._create_loader(
                Subset(dataset, train_idx),
                shuffle=True
            )
            
            val_loader = self._create_loader(
                Subset(dataset, val_idx),
                shuffle=False
            )
            
            yield fold, train_loader, val_loader
    
    def _create_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None
        )


class DataAugmentation:
    
    @staticmethod
    def add_noise(X: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        noise = np.random.normal(0, noise_level, X.shape)
        return X + noise
    
    @staticmethod
    def mixup(X: np.ndarray, y: np.ndarray, alpha: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        n = len(X)
        indices = np.random.permutation(n)
        
        lam = np.random.beta(alpha, alpha, n)
        lam = lam.reshape(-1, 1)
        
        X_mixed = lam * X + (1 - lam) * X[indices]
        y_mixed = lam.squeeze() * y + (1 - lam.squeeze()) * y[indices]
        
        return X_mixed, y_mixed
    
    @staticmethod
    def cutout(X: np.ndarray, n_features_to_mask: int = 3) -> np.ndarray:
        X_aug = X.copy()
        n_samples, n_features = X.shape
        
        for i in range(n_samples):
            mask_indices = np.random.choice(n_features, n_features_to_mask, replace=False)
            X_aug[i, mask_indices] = 0
        
        return X_aug


class DataValidator:
    
    @staticmethod
    def validate_shape(X: np.ndarray, y: np.ndarray) -> None:
        assert len(X.shape) == 2, "X must be 2D"
        assert len(y.shape) == 1, "y must be 1D"
        assert X.shape[0] == y.shape[0], "X and y must have same length"
    
    @staticmethod
    def validate_values(X: np.ndarray, y: np.ndarray) -> None:
        assert not np.any(np.isnan(X)), "X contains NaN"
        assert not np.any(np.isinf(X)), "X contains inf"
        assert not np.any(np.isnan(y)), "y contains NaN"
        assert not np.any(np.isinf(y)), "y contains inf"
        
    @staticmethod
    def validate_range(y: np.ndarray, min_val: float = 0, max_val: float = 500) -> None:
        valid_y = y[~np.isnan(y) & ~np.isinf(y)]
        if len(valid_y) > 0:
            assert np.all(valid_y >= min_val), f"y contains values below {min_val}: min={valid_y.min()}"
            assert np.all(valid_y <= max_val), f"y contains values above {max_val}: max={valid_y.max()}"
    
    @staticmethod
    def validate_all(X: np.ndarray, y: np.ndarray) -> None:
        DataValidator.validate_shape(X, y)
        DataValidator.validate_values(X, y)
        DataValidator.validate_range(y)


class DataCache:
    
    def __init__(self, cache_dir: str = ".cache/data"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, key: str, data: Dict) -> None:
        cache_file = self.cache_dir / f"{key}.pkl"
        joblib.dump(data, cache_file, compress=3)
    
    def load(self, key: str) -> Optional[Dict]:
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            return joblib.load(cache_file)
        return None
    
    def exists(self, key: str) -> bool:
        cache_file = self.cache_dir / f"{key}.pkl"
        return cache_file.exists()
    
    def clear(self) -> None:
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
