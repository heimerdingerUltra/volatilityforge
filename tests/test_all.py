import unittest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import ModelType, create_model_config, create_training_config
from src.data.advanced_features import AdvancedFeatures
from src.data.pipeline import DataPipeline, DataValidator
from src.models.ensemble import ModelFactory
from src.models.advanced_ensemble import AdvancedEnsembleSystem
from src.evaluation.metrics import MetricsCalculator


class TestFeatureEngineering(unittest.TestCase):
    
    def setUp(self):
        self.sample_data = pd.DataFrame({
            'BID': [10.0, 11.0, 12.0],
            'ASK': [10.5, 11.5, 12.5],
            'BIDSIZE': [100, 200, 150],
            'ASKSIZE': [150, 180, 200],
            'STRIKE_PRC': [100, 105, 110],
            'DAYS_TO_EXPIRY_CALC': [30, 60, 90],
            'ACVOL_1': [1000, 1500, 2000],
            'OPINT_1': [5000, 6000, 7000],
            'OPTION_TYPE': ['CALL', 'PUT', 'CALL'],
            'IMP_VOLT': [20.0, 25.0, 30.0]
        })
    
    def test_feature_extraction(self):
        fe = AdvancedFeatures(cache_dir='.cache/test')
        X, y = fe.fit_transform(self.sample_data, use_cache=False)
        
        self.assertEqual(len(X), 3)
        self.assertTrue(fe.feature_names is not None)
        self.assertTrue(len(fe.feature_names) > 10)
    
    def test_microstructure_features(self):
        from src.data.advanced_features import QuantitativeFeatureEngine
        
        features = QuantitativeFeatureEngine.microstructure_features(self.sample_data)
        
        self.assertIn('mid_price', features.columns)
        self.assertIn('spread', features.columns)
        self.assertIn('order_imbalance', features.columns)
    
    def test_data_validation(self):
        X = np.random.randn(100, 20)
        y = np.abs(np.random.randn(100)) * 10 + 20
        
        DataValidator.validate_all(X, y)
        
        with self.assertRaises(AssertionError):
            y_invalid = np.array([np.nan] * 100)
            DataValidator.validate_all(X, y_invalid)


class TestDataPipeline(unittest.TestCase):
    
    def setUp(self):
        self.X = np.random.randn(1000, 20)
        self.y = np.random.randn(1000) * 10 + 20
    
    def test_create_loaders(self):
        pipeline = DataPipeline(batch_size=32, num_workers=0)
        
        train_loader, val_loader, test_loader = pipeline.create_loaders(
            self.X, self.y,
            test_size=0.2,
            val_size=0.1,
            seed=42
        )
        
        self.assertTrue(len(train_loader) > 0)
        self.assertTrue(len(val_loader) > 0)
        self.assertTrue(len(test_loader) > 0)
    
    def test_kfold_loaders(self):
        pipeline = DataPipeline(batch_size=32, num_workers=0)
        
        folds = list(pipeline.create_kfold_loaders(
            self.X, self.y,
            n_splits=3,
            seed=42
        ))
        
        self.assertEqual(len(folds), 3)


class TestModels(unittest.TestCase):
    
    def setUp(self):
        self.n_features = 20
        self.batch_size = 16
        self.X = torch.randn(self.batch_size, self.n_features)
    
    def test_tabpfn(self):
        config = create_model_config(ModelType.TABPFN, self.n_features)
        model = ModelFactory.create_tabpfn(self.n_features, config.hyperparameters)
        
        output = model(self.X)
        
        self.assertEqual(output.shape, (self.batch_size,))
    
    def test_mamba(self):
        config = create_model_config(ModelType.MAMBA, self.n_features)
        model = ModelFactory.create_mamba(self.n_features, config.hyperparameters)
        
        output = model(self.X)
        
        self.assertEqual(output.shape, (self.batch_size,))
    
    def test_xlstm(self):
        config = create_model_config(ModelType.XLSTM, self.n_features)
        model = ModelFactory.create_xlstm(self.n_features, config.hyperparameters)
        
        output = model(self.X)
        
        self.assertEqual(output.shape, (self.batch_size,))


class TestEnsemble(unittest.TestCase):
    
    def setUp(self):
        self.n_features = 20
        self.batch_size = 16
        
        self.models = {
            'model1': ModelFactory.create_tabpfn(self.n_features, {'d_model': 64, 'n_layers': 2, 'n_heads': 4}),
            'model2': ModelFactory.create_mamba(self.n_features, {'d_model': 64, 'n_layers': 2}),
        }
        
        self.X = torch.randn(self.batch_size, self.n_features)
    
    def test_attention_ensemble(self):
        ensemble = AdvancedEnsembleSystem(
            self.models,
            device='cpu',
            ensemble_type='attention'
        )
        
        output = ensemble.forward(self.X)
        
        self.assertEqual(output.shape, (self.batch_size,))
    
    def test_hierarchical_ensemble(self):
        ensemble = AdvancedEnsembleSystem(
            self.models,
            device='cpu',
            ensemble_type='hierarchical'
        )
        
        output = ensemble.forward(self.X)
        
        self.assertEqual(output.shape, (self.batch_size,))
    
    def test_uncertainty_ensemble(self):
        ensemble = AdvancedEnsembleSystem(
            self.models,
            device='cpu',
            ensemble_type='uncertainty'
        )
        
        mean, uncertainty = ensemble.forward(self.X, return_uncertainty=True)
        
        self.assertEqual(mean.shape, (self.batch_size,))
        self.assertEqual(uncertainty.shape, (self.batch_size,))


class TestMetrics(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        self.y_true = np.random.randn(100) * 10 + 20
        self.y_pred = self.y_true + np.random.randn(100) * 2
    
    def test_regression_metrics(self):
        metrics = MetricsCalculator.compute_regression_metrics(
            self.y_true,
            self.y_pred
        )
        
        self.assertTrue(metrics.rmse > 0)
        self.assertTrue(metrics.mae > 0)
        self.assertTrue(0 <= metrics.r2 <= 1)
    
    def test_quantile_metrics(self):
        metrics = MetricsCalculator.compute_quantile_metrics(
            self.y_true,
            self.y_pred
        )
        
        self.assertIn('error_q50', metrics)
        self.assertIn('error_q90', metrics)
    
    def test_directional_accuracy(self):
        metrics = MetricsCalculator.compute_directional_accuracy(
            self.y_true,
            self.y_pred
        )
        
        self.assertIn('accuracy', metrics)
        self.assertTrue(0 <= metrics['accuracy'] <= 1)


class TestConfiguration(unittest.TestCase):
    
    def test_training_config_creation(self):
        config = create_training_config(strategy='balanced')
        
        self.assertEqual(config.batch_size, 512)
        self.assertEqual(config.epochs, 200)
        self.assertTrue(config.mixed_precision)
    
    def test_model_config_creation(self):
        config = create_model_config(
            ModelType.TABPFN,
            n_features=20,
            d_model=256
        )
        
        self.assertEqual(config.model_type, ModelType.TABPFN)
        self.assertEqual(config.n_features, 20)
        self.assertEqual(config.hyperparameters['d_model'], 256)


class TestModelRegistry(unittest.TestCase):
    
    def setUp(self):
        from src.models.registry import ModelRegistry
        import tempfile
        
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ModelRegistry(self.temp_dir)
    
    def test_register_and_load(self):
        from src.models.registry import ModelMetadata, create_version_string
        
        n_features = 20
        model = ModelFactory.create_tabpfn(n_features, {'d_model': 64, 'n_layers': 2, 'n_heads': 4})
        
        metadata = ModelMetadata(
            model_name='test_model',
            model_type='tabpfn',
            version=create_version_string(),
            timestamp='2025-01-01T00:00:00',
            n_features=n_features,
            hyperparameters={'d_model': 64},
            metrics={'rmse': 1.5}
        )
        
        model_key = self.registry.register_model(model, metadata)
        
        self.assertTrue(model_key in self.registry.list_models())


def run_tests():
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
