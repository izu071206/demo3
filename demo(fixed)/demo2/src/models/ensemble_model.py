"""
Ensemble Model - Kết hợp Random Forest, XGBoost, Neural Network
File: src/models/ensemble_model.py
"""

import numpy as np
import joblib
from typing import List, Dict, Optional
import logging
from pathlib import Path

from .base_model import BaseModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel
try:
    from .neural_network_model import NeuralNetworkModel
    NN_AVAILABLE = True
except (ImportError, OSError):
    NN_AVAILABLE = False

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """
    Ensemble model kết hợp nhiều classifiers
    Strategies: voting, weighted_voting, stacking
    """
    
    def __init__(self, strategy: str = 'weighted_voting', 
                 weights: Optional[List[float]] = None,
                 base_models: Optional[List[str]] = None):
        """
        Args:
            strategy: 'voting', 'weighted_voting', 'stacking'
            weights: Weights cho weighted voting [RF, XGB, NN]
            base_models: List of models to use ['rf', 'xgb', 'nn']
        """
        super().__init__("Ensemble")
        self.strategy = strategy
        self.base_models = base_models or ['rf', 'xgb']  # Default: RF + XGB
        
        # Default weights based on typical performance
        if weights is None:
            if 'nn' in self.base_models:
                self.weights = np.array([0.35, 0.35, 0.30])  # RF, XGB, NN
            else:
                self.weights = np.array([0.5, 0.5])  # RF, XGB
        else:
            self.weights = np.array(weights)
        
        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)
        
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize base models"""
        if 'rf' in self.base_models:
            self.models['rf'] = RandomForestModel(
                n_estimators=150,
                max_depth=25,
                min_samples_split=3,
                class_weight='balanced'
            )
        
        if 'xgb' in self.base_models:
            self.models['xgb'] = XGBoostModel(
                n_estimators=150,
                max_depth=8,
                learning_rate=0.05
            )
        
        if 'nn' in self.base_models and NN_AVAILABLE:
            self.models['nn'] = NeuralNetworkModel(
                hidden_layers=[256, 128, 64],
                dropout=0.4,
                epochs=100
            )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """Train all base models"""
        logger.info(f"Training Ensemble Model with strategy: {self.strategy}")
        logger.info(f"Base models: {list(self.models.keys())}")
        
        history = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name.upper()}...")
            model_history = model.train(X_train, y_train, X_val, y_val)
            history[name] = model_history
            logger.info(f"{name.upper()} training completed")
        
        self.is_trained = True
        
        # Calculate ensemble validation accuracy
        if X_val is not None and y_val is not None and len(X_val) > 0:
            ensemble_preds = self.predict(X_val)
            ensemble_acc = np.mean(ensemble_preds == y_val)
            history['ensemble_val_accuracy'] = ensemble_acc
            logger.info(f"Ensemble validation accuracy: {ensemble_acc:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Ensemble prediction"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if self.strategy == 'voting':
            return self._voting_predict(X)
        elif self.strategy == 'weighted_voting':
            return self._weighted_voting_predict(X)
        elif self.strategy == 'stacking':
            return self._stacking_predict(X)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Ensemble probability prediction"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        all_probas = []
        
        for name, model in self.models.items():
            probas = model.predict_proba(X)
            all_probas.append(probas)
        
        # Weighted average of probabilities
        ensemble_proba = np.zeros_like(all_probas[0])
        for i, proba in enumerate(all_probas):
            weight = self.weights[i] if i < len(self.weights) else 1.0/len(all_probas)
            ensemble_proba += weight * proba
        
        return ensemble_proba
    
    def _voting_predict(self, X: np.ndarray) -> np.ndarray:
        """Simple majority voting"""
        predictions = []
        
        for model in self.models.values():
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        # Majority vote for each sample
        return np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(),
            axis=0,
            arr=predictions
        )
    
    def _weighted_voting_predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted voting based on model confidence"""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def _stacking_predict(self, X: np.ndarray) -> np.ndarray:
        """Stacking prediction (use probabilities as meta-features)"""
        # For now, use weighted voting
        # TODO: Implement proper stacking with meta-learner
        return self._weighted_voting_predict(X)
    
    def get_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each base model"""
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X)
        return predictions
    
    def get_model_probabilities(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get probabilities from each base model"""
        probabilities = {}
        for name, model in self.models.items():
            probabilities[name] = model.predict_proba(X)
        return probabilities
    
    def save(self, filepath: str):
        """Save ensemble model"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        filepath = Path(filepath)
        base_path = filepath.parent / filepath.stem
        
        # Save each model
        for name, model in self.models.items():
            model_path = f"{base_path}_{name}{filepath.suffix}"
            model.save(model_path)
        
        # Save ensemble metadata
        metadata = {
            'strategy': self.strategy,
            'weights': self.weights.tolist(),
            'base_models': self.base_models,
            'model_paths': {
                name: f"{base_path}_{name}{filepath.suffix}"
                for name in self.models.keys()
            }
        }
        
        metadata_path = f"{base_path}_metadata.pkl"
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"Ensemble model saved to {base_path}_*")
    
    def load(self, filepath: str):
        """Load ensemble model"""
        filepath = Path(filepath)
        base_path = filepath.parent / filepath.stem
        
        # Load metadata
        metadata_path = f"{base_path}_metadata.pkl"
        metadata = joblib.load(metadata_path)
        
        self.strategy = metadata['strategy']
        self.weights = np.array(metadata['weights'])
        self.base_models = metadata['base_models']
        
        # Initialize and load each model
        self._initialize_models()
        for name, model_path in metadata['model_paths'].items():
            if name in self.models:
                self.models[name].load(model_path)
        
        self.is_trained = True
        logger.info(f"Ensemble model loaded from {base_path}_*")
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """Get feature importance from tree-based models"""
        importance = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'get_feature_importance'):
                importance[name] = model.get_feature_importance()
        
        return importance
    
    def analyze_disagreement(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict:
        """
        Analyze where models disagree
        Useful for identifying hard samples
        """
        model_preds = self.get_model_predictions(X)
        ensemble_pred = self.predict(X)
        
        # Find samples where models disagree
        pred_matrix = np.array(list(model_preds.values()))
        disagreement = []
        
        for i in range(X.shape[0]):
            sample_preds = pred_matrix[:, i]
            if len(np.unique(sample_preds)) > 1:  # Disagreement
                disagreement.append({
                    'index': i,
                    'predictions': {
                        name: int(pred[i]) 
                        for name, pred in model_preds.items()
                    },
                    'ensemble_pred': int(ensemble_pred[i]),
                    'true_label': int(y_true[i]) if y_true is not None else None
                })
        
        analysis = {
            'total_samples': X.shape[0],
            'disagreements': len(disagreement),
            'disagreement_rate': len(disagreement) / X.shape[0],
            'disagreement_details': disagreement[:10]  # Top 10
        }
        
        if y_true is not None:
            # Analyze if disagreements are on correct/incorrect predictions
            correct_disagree = sum(
                1 for d in disagreement 
                if d['ensemble_pred'] == d['true_label']
            )
            analysis['correct_on_disagreement'] = correct_disagree / len(disagreement) if disagreement else 0
        
        return analysis