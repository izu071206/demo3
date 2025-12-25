"""
Improved Dataset Generation với Advanced Features
File: src/dataset/generate_dataset_improved.py
"""
# python -m src.dataset.generate_dataset_improved --config config/dataset_config.yaml

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pefile
import yaml
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

# Import improved feature pipeline
from src.features.feature_pipeline_improved import (
    ImprovedFeaturePipeline, 
    ImprovedFeaturePipelineConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedDatasetGenerator:
    """Generate dataset với advanced features"""
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to dataset config YAML
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['dataset']
        
        # Load improved pipeline config
        pipeline_config = ImprovedFeaturePipelineConfig.from_dataset_config(config_path)
        self.feature_pipeline = ImprovedFeaturePipeline(pipeline_config)
        
        self.metadata_path = Path(self.config['processed_features_dir']) / 'feature_metadata.json'
        
        logger.info("="*60)
        logger.info("IMPROVED DATASET GENERATOR")
        logger.info("="*60)
        logger.info(f"Advanced features: {pipeline_config.enable_advanced_features}")
        logger.info(f"Entropy analysis: {pipeline_config.enable_entropy_analysis}")
        logger.info(f"Packer detection: {pipeline_config.enable_packer_detection}")
        logger.info(f"String analysis: {pipeline_config.enable_string_analysis}")
        logger.info(f"PE analysis: {pipeline_config.enable_pe_analysis}")
        logger.info("="*60)
    
    def extract_features_from_file(self, file_path: str) -> np.ndarray:
        """Extract all features from file"""
        return self.feature_pipeline.build_feature_vector(file_path)
    
    def is_valid_binary_file(self, file_path: Path) -> bool:
        """Check if file is valid binary"""
        # Skip non-binary files
        skip_extensions = {
            '.gitkeep', '.txt', '.md', '.py', '.yaml', '.yml',
            '.json', '.csv', '.pkl', '.pt', '.log', '.png', '.jpg'
        }
        
        if file_path.suffix.lower() in skip_extensions:
            return False
        
        # Skip hidden files
        if file_path.name.startswith('.'):
            return False
        
        # Check file size (at least 100 bytes)
        try:
            if file_path.stat().st_size < 100:
                return False
        except:
            return False
        
        # Check if binary
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(512)
                if len(chunk) == 0:
                    return False
                
                # Check for null bytes
                null_count = chunk.count(b'\x00')
                if null_count > 10:
                    return True
                
                # Try decode as text
                try:
                    chunk.decode('utf-8')
                    if null_count == 0:
                        return False
                except:
                    return True
        except:
            return False
        
        return True
    
    def validate_pe(self, file_path: Path) -> bool:
        """Validate PE file"""
        try:
            pe = pefile.PE(str(file_path), fast_load=True)
            is_valid = pe.DOS_HEADER.e_magic == 0x5A4D
            pe.close()
            return is_valid
        except Exception:
            return False
    
    def get_family_name(self, root_dir: Path, file_path: Path, label: int) -> str:
        """Get malware family from directory structure"""
        try:
            relative = file_path.relative_to(root_dir)
            parts = relative.parts
            if len(parts) > 1:
                return parts[0]
        except Exception:
            pass
        return file_path.parent.name or ("benign" if label == 0 else "malware")
    
    def process_directory(self, directory: str, label: int) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Process all files in directory with improved features
        
        Args:
            directory: Directory path
            label: Label (0: benign, 1: obfuscated)
        
        Returns:
            Tuple of (features, labels, metadata)
        """
        features_list: List[np.ndarray] = []
        labels_list: List[int] = []
        metadata_list: List[Dict] = []
        
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return np.array([]), np.array([]), []
        
        # Get all valid files
        all_files = list(Path(directory).rglob('*'))
        files = [f for f in all_files if f.is_file() and self.is_valid_binary_file(f)]
        
        if len(files) == 0:
            logger.warning(f"No valid binary files found in {directory}")
            return np.array([]), np.array([]), []
        
        logger.info(f"Processing {len(files)} valid binary files from {directory}")
        
        root_dir = Path(directory)
        successful = 0
        failed = 0
        
        for file_path in tqdm(files, desc=f"Processing {directory}"):
            try:
                # Validate PE
                if not self.validate_pe(file_path):
                    logger.debug(f"Skipping invalid PE file: {file_path.name}")
                    failed += 1
                    continue
                
                # Extract features
                features = self.extract_features_from_file(str(file_path))
                
                if len(features) > 0:
                    features_list.append(features)
                    labels_list.append(label)
                    
                    metadata_list.append({
                        'file_path': str(file_path),
                        'label': label,
                        'family': self.get_family_name(root_dir, file_path, label),
                        'source_dir': directory,
                        'size_bytes': file_path.stat().st_size,
                        'feature_dim': len(features)
                    })
                    
                    successful += 1
                else:
                    logger.debug(f"No features extracted from {file_path.name}")
                    failed += 1
            
            except Exception as e:
                logger.warning(f"Error processing {file_path}: {e}")
                failed += 1
        
        logger.info(f"  Successful: {successful}, Failed: {failed}")
        
        if features_list:
            max_len = max(len(f) for f in features_list)
            padded_features = [
                self.feature_pipeline.pad_vector(f, max_len)
                for f in features_list
            ]
            return np.array(padded_features), np.array(labels_list), metadata_list
        else:
            return np.array([]), np.array([]), []
    
    def pad_matrix(self, matrix: np.ndarray, target_dim: int) -> np.ndarray:
        """Pad or truncate matrix to target dimension"""
        if matrix.size == 0:
            return matrix
        
        current_dim = matrix.shape[1]
        if current_dim == target_dim:
            return matrix
        
        if current_dim < target_dim:
            padding = np.zeros((matrix.shape[0], target_dim - current_dim))
            return np.hstack([matrix, padding])
        
        return matrix[:, :target_dim]
    
    def save_split(self, output_dir: str, name: str, features: np.ndarray, labels: np.ndarray):
        """Save data split"""
        with open(os.path.join(output_dir, f'{name}_features.pkl'), 'wb') as f:
            pickle.dump((features, labels), f)
        logger.info(f"  Saved {name}: {features.shape}")
    
    def generate_dataset(self):
        """Generate complete dataset with advanced features"""
        logger.info("Starting improved dataset generation...")
        
        # 1. Process benign samples
        benign_dir = self.config['benign_source_dir']
        logger.info(f"\n[1/2] Processing benign samples from {benign_dir}")
        benign_features, benign_labels, benign_metadata = self.process_directory(benign_dir, label=0)
        
        # 2. Process obfuscated samples
        obfuscated_dir = self.config['obfuscated_output_dir']
        logger.info(f"\n[2/2] Processing obfuscated samples from {obfuscated_dir}")
        obfuscated_features, obfuscated_labels, obfuscated_metadata = self.process_directory(
            obfuscated_dir, label=1
        )
        
        # 3. Combine datasets
        available_sets = [arr for arr in [benign_features, obfuscated_features] if arr.size > 0]
        if not available_sets:
            logger.error("No features extracted! Please add binary samples.")
            return
        
        # Determine target dimension
        target_dim = max(arr.shape[1] for arr in available_sets)
        logger.info(f"\nTarget feature dimension: {target_dim}")
        
        # Pad to same dimension
        benign_features = self.pad_matrix(benign_features, target_dim)
        obfuscated_features = self.pad_matrix(obfuscated_features, target_dim)
        
        # Combine
        if len(benign_features) > 0 and len(obfuscated_features) > 0:
            all_features = np.vstack([benign_features, obfuscated_features])
            all_labels = np.hstack([benign_labels, obfuscated_labels])
            all_metadata = benign_metadata + obfuscated_metadata
        elif len(benign_features) > 0:
            all_features = benign_features
            all_labels = benign_labels
            all_metadata = benign_metadata
        elif len(obfuscated_features) > 0:
            all_features = obfuscated_features
            all_labels = obfuscated_labels
            all_metadata = obfuscated_metadata
        else:
            logger.error("No features to process!")
            return
        
        logger.info(f"\nCombined dataset: {all_features.shape}")
        logger.info(f"  Benign: {np.sum(all_labels == 0)}")
        logger.info(f"  Obfuscated: {np.sum(all_labels == 1)}")
        
        # 4. Split dataset by family (to avoid data leakage)
        groups = np.array([meta.get('family', 'unknown') for meta in all_metadata])
        unique_families = np.unique(groups)
        logger.info(f"\nFound {len(unique_families)} families: {list(unique_families)}")
        
        rng = self.config.get('random_state', 42)
        test_ratio = self.config.get('test_ratio', 0.15)
        val_ratio = self.config.get('val_ratio', 0.15)
        train_ratio = self.config.get('train_ratio', 0.7)
        
        def group_split(features, labels, groups, test_size, random_state):
            """Split by groups to avoid leakage"""
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            try:
                train_idx, test_idx = next(splitter.split(features, labels, groups=groups))
                return train_idx, test_idx
            except ValueError as exc:
                logger.warning(f"Group split failed ({exc}). Using random split.")
                indices = np.random.permutation(len(features))
                cutoff = int(len(features) * (1 - test_size))
                return indices[:cutoff], indices[cutoff:]
        
        # Split train+val / test
        trainval_idx, test_idx = group_split(all_features, all_labels, groups, test_ratio, rng)
        
        # Split train / val
        remaining_ratio = train_ratio + val_ratio
        val_relative = val_ratio / remaining_ratio if remaining_ratio > 0 else 0.0
        
        trainval_features = all_features[trainval_idx]
        trainval_labels = all_labels[trainval_idx]
        trainval_groups = groups[trainval_idx]
        
        train_idx_rel, val_idx_rel = group_split(
            trainval_features, trainval_labels, trainval_groups, 
            val_relative, rng + 1
        )
        
        train_idx = trainval_idx[train_idx_rel]
        val_idx = trainval_idx[val_idx_rel]
        
        # Final splits
        X_train, y_train = all_features[train_idx], all_labels[train_idx]
        X_val, y_val = all_features[val_idx], all_labels[val_idx]
        X_test, y_test = all_features[test_idx], all_labels[test_idx]
        
        # Create split map for metadata
        split_map = ['train'] * len(all_features)
        for idx in val_idx:
            split_map[idx] = 'val'
        for idx in test_idx:
            split_map[idx] = 'test'
        
        # 5. Save splits
        output_dir = self.config['processed_features_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("\nSaving data splits...")
        self.save_split(output_dir, 'train', X_train, y_train)
        self.save_split(output_dir, 'val', X_val, y_val)
        self.save_split(output_dir, 'test', X_test, y_test)
        
        # 6. Save metadata
        metadata_df = pd.DataFrame(all_metadata)
        metadata_df['split'] = split_map
        metadata_path = Path(output_dir) / 'sample_metadata.csv'
        metadata_df.to_csv(metadata_path, index=False)
        logger.info(f"  Saved metadata: {metadata_path}")
        
        # 7. Save feature metadata
        self.feature_pipeline.save_metadata(self.metadata_path, int(target_dim))
        
        # 8. Print summary
        logger.info("\n" + "="*60)
        logger.info("DATASET GENERATION COMPLETED")
        logger.info("="*60)
        logger.info(f"Train: {X_train.shape[0]} samples ({np.sum(y_train==0)} benign, {np.sum(y_train==1)} obfuscated)")
        logger.info(f"Val:   {X_val.shape[0]} samples ({np.sum(y_val==0)} benign, {np.sum(y_val==1)} obfuscated)")
        logger.info(f"Test:  {X_test.shape[0]} samples ({np.sum(y_test==0)} benign, {np.sum(y_test==1)} obfuscated)")
        logger.info(f"Feature dimension: {target_dim}")
        logger.info(f"Metadata saved: {metadata_path}")
        logger.info("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate improved dataset")
    parser.add_argument("--config", type=str, default="config/dataset_config.yaml",
                       help="Path to dataset config file")
    
    args = parser.parse_args()
    
    generator = ImprovedDatasetGenerator(args.config)
    generator.generate_dataset()