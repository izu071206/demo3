"""
Malware Family Classifier
Nhận diện loại/family của malware
File: src/models/family_classifier.py
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MalwareFamilyClassifier:
    """
    Multi-class classifier để nhận diện malware family
    Kết hợp với obfuscation detection
    """
    
    def __init__(self, n_estimators: int = 200, max_depth: int = 30):
        """
        Args:
            n_estimators: Number of trees
            max_depth: Maximum depth
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.family_names = []
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict:
        """
        Train family classifier
        
        Args:
            X_train: Features
            y_train: Family labels (strings)
            X_val: Validation features
            y_val: Validation labels
        """
        logger.info("Training Malware Family Classifier...")
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.family_names = self.label_encoder.classes_.tolist()
        
        logger.info(f"Found {len(self.family_names)} malware families:")
        for i, name in enumerate(self.family_names):
            count = np.sum(y_train_encoded == i)
            logger.info(f"  {name}: {count} samples")
        
        # Train
        self.model.fit(X_train, y_train_encoded)
        self.is_trained = True
        
        # Evaluate
        train_acc = self.model.score(X_train, y_train_encoded)
        
        history = {
            'train_accuracy': train_acc,
            'num_families': len(self.family_names),
            'families': self.family_names
        }
        
        if X_val is not None and y_val is not None and len(X_val) > 0:
            y_val_encoded = self.label_encoder.transform(y_val)
            val_acc = self.model.score(X_val, y_val_encoded)
            history['val_accuracy'] = val_acc
            logger.info(f"Validation accuracy: {val_acc:.4f}")
        
        logger.info(f"Training accuracy: {train_acc:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict family labels
        
        Returns:
            Array of family names
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for each family
        
        Returns:
            Array of probabilities [n_samples, n_families]
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        return self.model.predict_proba(X)
    
    def predict_with_confidence(self, X: np.ndarray, 
                                threshold: float = 0.5) -> List[Dict]:
        """
        Predict với confidence score và top-k families
        
        Args:
            X: Features
            threshold: Confidence threshold
            
        Returns:
            List of predictions with details
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        probas = self.predict_proba(X)
        predictions = []
        
        for i, proba in enumerate(probas):
            # Sort by probability
            sorted_indices = np.argsort(proba)[::-1]
            
            top_family = self.family_names[sorted_indices[0]]
            top_confidence = proba[sorted_indices[0]]
            
            # Get top 3 predictions
            top_3 = [
                {
                    'family': self.family_names[idx],
                    'confidence': float(proba[idx])
                }
                for idx in sorted_indices[:3]
            ]
            
            prediction = {
                'family': top_family,
                'confidence': float(top_confidence),
                'top_3_families': top_3,
                'is_confident': top_confidence >= threshold
            }
            
            predictions.append(prediction)
        
        return predictions
    
    def get_family_info(self) -> Dict[str, Dict]:
        """
        Get information about each malware family
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        info = {}
        
        for family in self.family_names:
            info[family] = {
                'name': family,
                'description': self._get_family_description(family),
                'severity': self._get_family_severity(family),
                'common_behaviors': self._get_family_behaviors(family)
            }
        
        return info
    
    def _get_family_description(self, family: str) -> str:
        """Get description for malware family"""
        # Database of known families (can be expanded)
        descriptions = {
            'emotet': 'Banking Trojan and malware downloader',
            'trickbot': 'Banking Trojan with info-stealing capabilities',
            'ransomware': 'Encrypts files and demands ransom',
            'botnet': 'Part of a botnet network for DDoS or spam',
            'trojan': 'General trojan malware',
            'worm': 'Self-replicating malware',
            'backdoor': 'Provides unauthorized remote access',
            'spyware': 'Steals sensitive information',
            'adware': 'Displays unwanted advertisements',
            'downloader': 'Downloads and executes additional malware',
            'dropper': 'Drops and installs other malware',
            'rootkit': 'Hides malicious software from detection',
            'bashlite': 'IoT botnet malware',
            'mirai': 'IoT botnet malware for DDoS attacks',
            'zeus': 'Banking Trojan',
            'cryptolocker': 'Ransomware that encrypts files'
        }
        
        family_lower = family.lower()
        for key, desc in descriptions.items():
            if key in family_lower:
                return desc
        
        return 'Unknown malware family'
    
    def _get_family_severity(self, family: str) -> str:
        """Get severity level"""
        high_severity = ['ransomware', 'rootkit', 'backdoor', 'emotet', 'trickbot']
        medium_severity = ['trojan', 'worm', 'botnet', 'spyware']
        
        family_lower = family.lower()
        
        for keyword in high_severity:
            if keyword in family_lower:
                return 'HIGH'
        
        for keyword in medium_severity:
            if keyword in family_lower:
                return 'MEDIUM'
        
        return 'LOW'
    
    def _get_family_behaviors(self, family: str) -> List[str]:
        """Get common behaviors"""
        behaviors_map = {
            'ransomware': ['File encryption', 'Ransom demand', 'File deletion'],
            'trojan': ['Data theft', 'Backdoor installation', 'Remote access'],
            'botnet': ['DDoS attacks', 'Spam sending', 'Cryptocurrency mining'],
            'spyware': ['Keylogging', 'Screen capture', 'Data exfiltration'],
            'worm': ['Self-replication', 'Network spreading', 'System exploitation'],
            'backdoor': ['Remote access', 'Command execution', 'Data exfiltration'],
            'rootkit': ['Process hiding', 'File hiding', 'Registry manipulation']
        }
        
        family_lower = family.lower()
        for key, behaviors in behaviors_map.items():
            if key in family_lower:
                return behaviors
        
        return ['Unknown behaviors']
    
    def save(self, filepath: str):
        """Save classifier"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'family_names': self.family_names
        }, filepath)
        
        logger.info(f"Family classifier saved to {filepath}")
    
    def load(self, filepath: str):
        """Load classifier"""
        data = joblib.load(filepath)
        
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.family_names = data['family_names']
        self.is_trained = True
        
        logger.info(f"Family classifier loaded from {filepath}")
        logger.info(f"Loaded {len(self.family_names)} families")


class CombinedObfuscationFamilyModel:
    """
    Kết hợp Obfuscation Detection và Family Classification
    """
    
    def __init__(self, obfuscation_model, family_classifier):
        """
        Args:
            obfuscation_model: Trained obfuscation detection model
            family_classifier: Trained family classifier
        """
        self.obfuscation_model = obfuscation_model
        self.family_classifier = family_classifier
    
    def predict(self, X: np.ndarray) -> List[Dict]:
        """
        Predict both obfuscation and family
        
        Returns:
            List of combined predictions
        """
        # Obfuscation detection
        obf_predictions = self.obfuscation_model.predict(X)
        obf_probas = self.obfuscation_model.predict_proba(X)
        
        # Family classification
        family_predictions = self.family_classifier.predict_with_confidence(X)
        
        # Combine results
        results = []
        
        for i in range(len(X)):
            is_obfuscated = obf_predictions[i] == 1
            obf_confidence = float(obf_probas[i][1] if is_obfuscated else obf_probas[i][0])
            
            result = {
                'is_obfuscated': is_obfuscated,
                'obfuscation_confidence': obf_confidence,
                'family': family_predictions[i]['family'],
                'family_confidence': family_predictions[i]['confidence'],
                'top_families': family_predictions[i]['top_3_families'],
                'severity': self.family_classifier._get_family_severity(
                    family_predictions[i]['family']
                ),
                'description': self.family_classifier._get_family_description(
                    family_predictions[i]['family']
                )
            }
            
            results.append(result)
        
        return results
    
    def analyze_file(self, X: np.ndarray) -> Dict:
        """
        Comprehensive analysis of a file
        """
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        result = self.predict(X)[0]
        
        # Add risk assessment
        risk_score = 0.0
        
        if result['is_obfuscated']:
            risk_score += 0.4
        
        severity_scores = {'HIGH': 0.6, 'MEDIUM': 0.3, 'LOW': 0.1}
        risk_score += severity_scores.get(result['severity'], 0.1)
        
        result['risk_score'] = min(risk_score, 1.0)
        result['risk_level'] = (
            'CRITICAL' if risk_score > 0.8 else
            'HIGH' if risk_score > 0.6 else
            'MEDIUM' if risk_score > 0.4 else
            'LOW'
        )
        
        return result