"""
Improved Feature Pipeline với Advanced Features
File: src/features/feature_pipeline_improved.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .feature_combiner import FeatureCombiner
from .static import APIExtractor, CFGExtractor, OpcodeExtractor
from .static.advanced_features import AdvancedFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class ImprovedFeaturePipelineConfig:
    """Configuration for improved feature extraction"""
    
    # Existing configs
    opcode_max_features: int = 1000
    opcode_ngrams: Optional[list] = None
    api_max_features: int = 500
    api_list_path: Optional[str] = None
    enable_cfg: bool = True
    
    # New advanced feature configs
    enable_advanced_features: bool = True
    enable_entropy_analysis: bool = True
    enable_packer_detection: bool = True
    enable_string_analysis: bool = True
    enable_pe_analysis: bool = True
    
    @classmethod
    def from_dataset_config(cls, config_path: str) -> "ImprovedFeaturePipelineConfig":
        """Load from existing dataset config"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset config not found: {config_path}")
        
        with path.open("r", encoding="utf-8") as fh:
            import yaml
            raw_cfg = yaml.safe_load(fh)
        
        dataset_cfg = raw_cfg.get("dataset", raw_cfg)
        feature_cfg = dataset_cfg.get("features", {})
        opcode_cfg = feature_cfg.get("opcode_ngrams", {})
        api_cfg = feature_cfg.get("api_calls", {})
        cfg_cfg = feature_cfg.get("cfg", {})
        
        # Get advanced features config if exists
        advanced_cfg = feature_cfg.get("advanced", {})
        
        return cls(
            opcode_max_features=opcode_cfg.get("max_features", 1000),
            opcode_ngrams=opcode_cfg.get("n", [2, 3, 4]),
            api_max_features=api_cfg.get("max_features", 500),
            api_list_path=api_cfg.get("api_list_path"),
            enable_cfg=cfg_cfg.get("extract_metrics", True),
            enable_advanced_features=advanced_cfg.get("enabled", True),
            enable_entropy_analysis=advanced_cfg.get("entropy", True),
            enable_packer_detection=advanced_cfg.get("packer_detection", True),
            enable_string_analysis=advanced_cfg.get("string_analysis", True),
            enable_pe_analysis=advanced_cfg.get("pe_analysis", True),
        )
    
    @classmethod
    def from_metadata(cls, metadata_path: str):
        """Load from feature metadata"""
        path = Path(metadata_path)
        if not path.exists():
            raise FileNotFoundError(f"Feature metadata not found: {metadata_path}")
        
        with path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        
        config = cls(
            opcode_max_features=metadata.get("opcode_max_features", 1000),
            opcode_ngrams=metadata.get("opcode_ngrams", [2, 3, 4]),
            api_max_features=metadata.get("api_max_features", 500),
            api_list_path=metadata.get("api_list_path"),
            enable_cfg=metadata.get("enable_cfg", True),
            enable_advanced_features=metadata.get("enable_advanced_features", True),
            enable_entropy_analysis=metadata.get("enable_entropy_analysis", True),
            enable_packer_detection=metadata.get("enable_packer_detection", True),
            enable_string_analysis=metadata.get("enable_string_analysis", True),
            enable_pe_analysis=metadata.get("enable_pe_analysis", True),
        )
        
        feature_dim = int(metadata.get("feature_dim", 0))
        return config, feature_dim


class ImprovedFeaturePipeline:
    """
    Improved pipeline với advanced features
    """
    
    def __init__(self, config: ImprovedFeaturePipelineConfig):
        self.config = config
        self.expected_dim: Optional[int] = None
        
        # Basic extractors
        self.opcode_extractor = OpcodeExtractor(n_grams=config.opcode_ngrams or [2, 3, 4])
        self.cfg_extractor = CFGExtractor() if config.enable_cfg else None
        self.api_extractor = APIExtractor(api_list_path=config.api_list_path)
        
        # Advanced extractor
        self.advanced_extractor = (
            AdvancedFeatureExtractor() 
            if config.enable_advanced_features 
            else None
        )
        
        self.combiner = FeatureCombiner()
    
    @classmethod
    def from_metadata(cls, metadata_path: str) -> "ImprovedFeaturePipeline":
        """Load from metadata"""
        config, feature_dim = ImprovedFeaturePipelineConfig.from_metadata(metadata_path)
        pipeline = cls(config)
        pipeline.expected_dim = feature_dim
        return pipeline
    
    def extract_feature_dict(self, file_path: str) -> Dict[str, np.ndarray]:
        """Extract all features as dictionary"""
        aggregated: Dict[str, np.ndarray] = {}
        
        # 1. Basic Opcode features
        try:
            opcode_features = self.opcode_extractor.extract_from_file(
                file_path, max_features=self.config.opcode_max_features
            )
            aggregated.update(opcode_features)
            logger.debug(f"Extracted {len(opcode_features)} opcode features")
        except Exception as exc:
            logger.warning("Opcode extraction failed for %s: %s", file_path, exc)
        
        # 2. CFG features
        if self.cfg_extractor is not None:
            try:
                cfg_features = self.cfg_extractor.extract_features(file_path)
                aggregated.update(cfg_features)
                logger.debug(f"Extracted {len(cfg_features)} CFG features")
            except Exception as exc:
                logger.warning("CFG extraction failed for %s: %s", file_path, exc)
        
        # 3. API call features
        try:
            api_features = self.api_extractor.extract_api_features(
                file_path, max_features=self.config.api_max_features
            )
            aggregated.update(api_features)
            logger.debug(f"Extracted {len(api_features)} API features")
        except Exception as exc:
            logger.warning("API extraction failed for %s: %s", file_path, exc)
        
        # 4. Advanced features
        if self.advanced_extractor is not None:
            try:
                advanced_features = {}
                
                # Entropy analysis
                if self.config.enable_entropy_analysis:
                    entropy_feats = self.advanced_extractor.analyze_section_entropy(file_path)
                    advanced_features.update(entropy_feats)
                
                # Packer detection
                if self.config.enable_packer_detection:
                    packer_feats = self.advanced_extractor.detect_packer(file_path)
                    advanced_features.update(packer_feats)
                
                # String analysis
                if self.config.enable_string_analysis:
                    string_feats = self.advanced_extractor.analyze_strings(file_path)
                    advanced_features.update(string_feats)
                
                # PE structure
                if self.config.enable_pe_analysis:
                    pe_feats = self.advanced_extractor.analyze_pe_structure(file_path)
                    advanced_features.update(pe_feats)
                
                aggregated.update(advanced_features)
                logger.debug(f"Extracted {len(advanced_features)} advanced features")
                
            except Exception as exc:
                logger.warning("Advanced feature extraction failed for %s: %s", file_path, exc)
        
        return aggregated
    
    def build_feature_vector(self, file_path: str) -> np.ndarray:
        """Build combined feature vector"""
        feature_dict = self.extract_feature_dict(file_path)
        combined = self.combiner.combine(feature_dict)
        
        logger.info(f"Total features extracted: {len(combined)}")
        
        return combined
    
    def pad_vector(self, vector: np.ndarray, target_dim: int) -> np.ndarray:
        """Pad or truncate vector to target dimension"""
        if target_dim <= 0:
            return vector
        
        current = len(vector)
        if current == target_dim:
            return vector
        
        if current < target_dim:
            return np.pad(vector, (0, target_dim - current), mode="constant")
        
        return vector[:target_dim]
    
    def save_metadata(self, output_path: Path, feature_dim: int) -> None:
        """Save feature metadata"""
        metadata = {
            "feature_dim": int(feature_dim),
            "opcode_max_features": self.config.opcode_max_features,
            "opcode_ngrams": self.config.opcode_ngrams or [2, 3, 4],
            "api_max_features": self.config.api_max_features,
            "api_list_path": str(self.config.api_list_path) if self.config.api_list_path else None,
            "enable_cfg": self.config.enable_cfg,
            "enable_advanced_features": self.config.enable_advanced_features,
            "enable_entropy_analysis": self.config.enable_entropy_analysis,
            "enable_packer_detection": self.config.enable_packer_detection,
            "enable_string_analysis": self.config.enable_string_analysis,
            "enable_pe_analysis": self.config.enable_pe_analysis,
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        
        logger.info("Feature metadata saved to %s", output_path)