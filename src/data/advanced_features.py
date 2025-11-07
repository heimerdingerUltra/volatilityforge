import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
from pathlib import Path
import joblib
import hashlib
import json


class FeatureStore:
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = None
        self.feature_names = None
        self.feature_metadata = {}
        
    def _compute_hash(self, df: pd.DataFrame, config: Dict) -> str:
        config_str = json.dumps(config, sort_keys=True)
        data_hash = hashlib.md5(df.values.tobytes()).hexdigest()[:8]
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{data_hash}_{config_hash}"
    
    def _load_cache(self, cache_key: str) -> Optional[Tuple]:
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            return joblib.load(cache_file)
        return None
    
    def _save_cache(self, cache_key: str, data: Tuple):
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        joblib.dump(data, cache_file, compress=3)
    
    def get_scaler(self, scaler_type: str):
        scalers = {
            'robust': RobustScaler(),
            'standard': StandardScaler(),
            'quantile': QuantileTransformer(output_distribution='normal')
        }
        return scalers.get(scaler_type, RobustScaler())


class QuantitativeFeatureEngine:
    
    @staticmethod
    def microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        if 'BID' in df.columns and 'ASK' in df.columns:
            mid = (df['BID'] + df['ASK']) / 2
            spread = df['ASK'] - df['BID']
            
            features['mid_price'] = mid
            features['spread'] = spread
            features['spread_pct'] = spread / (mid + 1e-10)
            features['spread_bps'] = features['spread_pct'] * 10000
            features['log_mid'] = np.log1p(mid)
            features['relative_spread'] = spread / df['BID']
            
            if 'BIDSIZE' in df.columns and 'ASKSIZE' in df.columns:
                total = df['BIDSIZE'] + df['ASKSIZE']
                features['order_imbalance'] = (df['BIDSIZE'] - df['ASKSIZE']) / (total + 1e-10)
                features['bid_ratio'] = df['BIDSIZE'] / (total + 1e-10)
                features['ask_ratio'] = df['ASKSIZE'] / (total + 1e-10)
                features['order_flow_intensity'] = np.abs(features['order_imbalance'])
                features['weighted_mid'] = (
                    df['BID'] * df['ASKSIZE'] + df['ASK'] * df['BIDSIZE']
                ) / (total + 1e-10)
        
        return features
    
    @staticmethod
    def moneyness_features(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        strike = None
        for col in ['STRIKE_PRC', 'STRIKE_PRICE', 'STRIKE']:
            if col in df.columns:
                strike = df[col]
                break
        
        if strike is not None and 'mid_price' in features.columns:
            spot = features['mid_price']
            m = spot / (strike + 1e-10)
            
            features['moneyness'] = m
            features['log_moneyness'] = np.log(m + 1e-10)
            features['moneyness_sq'] = m ** 2
            features['moneyness_cube'] = m ** 3
            features['abs_log_moneyness'] = np.abs(features['log_moneyness'])
            features['atm_distance'] = np.abs(1 - m)
            
            features['itm_flag'] = (m > 1.0).astype(float)
            features['otm_flag'] = (m < 1.0).astype(float)
            features['atm_flag'] = (np.abs(features['log_moneyness']) < 0.05).astype(float)
            
            features['moneyness_category'] = pd.cut(
                m, bins=[0, 0.9, 0.95, 1.05, 1.1, np.inf],
                labels=[0, 1, 2, 3, 4]
            ).astype(float)
        
        return features
    
    @staticmethod
    def time_features(df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        if 'DAYS_TO_EXPIRY_CALC' in df.columns:
            tte = df['DAYS_TO_EXPIRY_CALC'] / 365.25
            tte_safe = np.maximum(tte, 1/365.25)
            
            features['tte'] = tte
            features['sqrt_tte'] = np.sqrt(np.maximum(tte, 0))
            features['log_tte'] = np.log(tte_safe)
            features['inv_tte'] = 1 / tte_safe
            features['tte_sq'] = tte ** 2
            
            features['short_term'] = (tte < 0.08).astype(float)
            features['medium_term'] = ((tte >= 0.08) & (tte < 0.25)).astype(float)
            features['long_term'] = (tte >= 0.25).astype(float)
            
            features['time_decay_factor'] = np.exp(-tte)
        
        return features
    
    @staticmethod
    def volume_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        
        if 'ACVOL_1' in df.columns:
            features['log_volume'] = np.log1p(df['ACVOL_1'])
            features['sqrt_volume'] = np.sqrt(df['ACVOL_1'])
        
        if 'OPINT_1' in df.columns:
            features['log_oi'] = np.log1p(df['OPINT_1'])
            features['sqrt_oi'] = np.sqrt(df['OPINT_1'])
        
        if 'ACVOL_1' in df.columns and 'OPINT_1' in df.columns:
            features['vol_oi_ratio'] = df['ACVOL_1'] / (df['OPINT_1'] + 1)
            features['turnover'] = df['ACVOL_1'] / (df['OPINT_1'] + 1)
            features['liquidity_score'] = np.log1p(df['ACVOL_1'] * df['OPINT_1'])
        
        return features
    
    @staticmethod
    def interaction_features(features: pd.DataFrame) -> pd.DataFrame:
        features = features.copy()
        
        if 'moneyness' in features.columns and 'tte' in features.columns:
            if 'moneyness_tte' not in features.columns:
                features['moneyness_tte'] = features['moneyness'] * features['tte']
            features['moneyness_sqrt_tte'] = features['moneyness'] * features['sqrt_tte']
            features['log_moneyness_log_tte'] = features['log_moneyness'] * features['log_tte']
            features['atm_distance_tte'] = features['atm_distance'] * features['tte']
        
        if 'spread_pct' in features.columns and 'tte' in features.columns:
            features['spread_tte'] = features['spread_pct'] * features['tte']
            features['spread_sqrt_tte'] = features['spread_pct'] * features['sqrt_tte']
        
        if 'log_volume' in features.columns and 'moneyness' in features.columns:
            features['volume_moneyness'] = features['log_volume'] * features['moneyness']
        
        if 'order_imbalance' in features.columns and 'spread_pct' in features.columns:
            features['imbalance_spread'] = features['order_imbalance'] * features['spread_pct']
        
        return features
    
    @staticmethod
    def advanced_features(df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        if 'moneyness' in features.columns and 'tte' in features.columns and 'spread_pct' in features.columns:
            features['effective_spread'] = features['spread_pct'] / (features['sqrt_tte'] + 1e-10)
            
        if 'log_volume' in features.columns and 'log_oi' in features.columns:
            features['activity_index'] = features['log_volume'] + features['log_oi']
        
        if 'moneyness' in features.columns:
            features['moneyness_momentum'] = features['moneyness'] - 1.0
            features['delta_proxy'] = 1 / (1 + np.exp(-5 * features['log_moneyness']))
            
        return features


class AdvancedFeatures(FeatureStore):
    
    def __init__(self, scaler_type: str = 'robust', cache_dir: str = ".cache"):
        super().__init__(cache_dir)
        self.scaler = self.get_scaler(scaler_type)
        self.engine = QuantitativeFeatureEngine()
        
    def fit_transform(
        self,
        df: pd.DataFrame,
        use_cache: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        config = {'scaler': str(type(self.scaler).__name__)}
        cache_key = self._compute_hash(df, config) if use_cache else None
        
        if use_cache and cache_key:
            cached = self._load_cache(cache_key)
            if cached is not None:
                X, y, self.scaler, self.feature_names = cached
                return X, y
        
        features = self._extract_features(df)
        
        if 'OPTION_TYPE' in df.columns:
            features['is_call'] = (df['OPTION_TYPE'] == 'CALL').astype(float)
        elif 'PUT_CALL' in df.columns:
            features['is_call'] = df['PUT_CALL'].astype(str).str.upper().str.contains('CALL', na=False).astype(float)
        
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median())
        features = features.fillna(0)
        
        self.feature_names = list(features.columns)
        self.feature_metadata = {
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names
        }
        
        X = self.scaler.fit_transform(features.values)
        
        if 'IMP_VOLT' not in df.columns:
            raise ValueError("IMP_VOLT not found")
        
        y = df['IMP_VOLT'].values
        valid = (y > 0) & (y < 500) & ~np.isnan(y)
        
        X_valid, y_valid = X[valid], y[valid]
        
        if use_cache and cache_key:
            self._save_cache(cache_key, (X_valid, y_valid, self.scaler, self.feature_names))
        
        return X_valid, y_valid
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        features = self._extract_features(df)
        
        if 'OPTION_TYPE' in df.columns:
            features['is_call'] = (df['OPTION_TYPE'] == 'CALL').astype(float)
        elif 'PUT_CALL' in df.columns:
            features['is_call'] = df['PUT_CALL'].astype(str).str.upper().str.contains('CALL', na=False).astype(float)
        
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median())
        features = features.fillna(0)
        
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0
        
        features = features[self.feature_names]
        
        return self.scaler.transform(features.values)
    
    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        if features.index.duplicated().any():
            features = features.reset_index(drop=True)
        
        micro = self.engine.microstructure_features(df)
        features = pd.concat([features, micro], axis=1)
        
        moneyness = self.engine.moneyness_features(df, features)
        features = pd.concat([features, moneyness], axis=1)
        
        time = self.engine.time_features(df)
        features = pd.concat([features, time], axis=1)
        
        volume = self.engine.volume_liquidity_features(df)
        features = pd.concat([features, volume], axis=1)
        
        features = self.engine.interaction_features(features)
        features = self.engine.advanced_features(df, features)
        
        return features

