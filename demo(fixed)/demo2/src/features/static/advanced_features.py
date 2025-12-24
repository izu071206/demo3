"""
Advanced Feature Extraction
- Entropy Analysis
- Packing Detection
- String Obfuscation Detection
- Section Analysis
File: src/features/static/advanced_features.py
"""

import math
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import pefile

import logging
logger = logging.getLogger(__name__)


class AdvancedFeatureExtractor:
    """Trích xuất advanced features cho obfuscation detection"""
    
    def __init__(self):
        # Packed/Protector signatures
        self.packer_signatures = {
            'UPX': [b'UPX0', b'UPX1', b'UPX2'],
            'PECompact': [b'PECompact2'],
            'ASPack': [b'ASPack', b'.aspack'],
            'FSG': [b'FSG!'],
            'MEW': [b'MEW'],
            'Petite': [b'petite'],
            'WWPack': [b'WWPack'],
            'Themida': [b'Themida', b'.themida'],
            'VMProtect': [b'VMProtect', b'.vmp'],
            'Armadillo': [b'Armadillo'],
        }
        
        # Suspicious section names
        self.suspicious_sections = {
            '.text', '.data', '.rdata', '.idata', '.edata', '.rsrc',
            '.reloc', '.bss', '.tls'
        }
    
    def calculate_entropy(self, data: bytes) -> float:
        """
        Tính Shannon entropy của data
        High entropy (>7.0) thường indicate compression/encryption
        """
        if not data:
            return 0.0
        
        # Count byte frequencies
        counter = Counter(data)
        length = len(data)
        
        # Calculate entropy
        entropy = 0.0
        for count in counter.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def analyze_section_entropy(self, binary_path: str) -> Dict[str, float]:
        """Phân tích entropy của từng section trong PE"""
        features = {
            'avg_section_entropy': 0.0,
            'max_section_entropy': 0.0,
            'min_section_entropy': 8.0,
            'high_entropy_sections': 0,  # sections with entropy > 7.0
            'suspicious_entropy_ratio': 0.0
        }
        
        try:
            pe = pefile.PE(binary_path)
            entropies = []
            high_entropy_count = 0
            
            for section in pe.sections:
                section_data = section.get_data()
                if len(section_data) > 0:
                    entropy = self.calculate_entropy(section_data)
                    entropies.append(entropy)
                    
                    if entropy > 7.0:
                        high_entropy_count += 1
            
            if entropies:
                features['avg_section_entropy'] = float(np.mean(entropies))
                features['max_section_entropy'] = float(np.max(entropies))
                features['min_section_entropy'] = float(np.min(entropies))
                features['high_entropy_sections'] = high_entropy_count
                features['suspicious_entropy_ratio'] = high_entropy_count / len(entropies)
            
            pe.close()
        
        except Exception as e:
            logger.warning(f"Error analyzing section entropy: {e}")
        
        return features
    
    def detect_packer(self, binary_path: str) -> Dict[str, float]:
        """
        Phát hiện packer/protector
        """
        features = {
            'is_packed': 0.0,
            'packer_confidence': 0.0,
            'known_packer': 0.0
        }
        
        try:
            with open(binary_path, 'rb') as f:
                data = f.read()
            
            # Check for known packer signatures
            for packer_name, signatures in self.packer_signatures.items():
                for sig in signatures:
                    if sig in data:
                        features['is_packed'] = 1.0
                        features['known_packer'] = 1.0
                        features['packer_confidence'] = 1.0
                        logger.info(f"Detected packer: {packer_name}")
                        return features
            
            # Heuristic packing detection
            pe = pefile.PE(binary_path)
            
            # Check 1: High entropy sections
            high_entropy_sections = 0
            for section in pe.sections:
                section_data = section.get_data()
                if len(section_data) > 0:
                    entropy = self.calculate_entropy(section_data)
                    if entropy > 7.0:
                        high_entropy_sections += 1
            
            # Check 2: Unusual section names
            unusual_sections = 0
            for section in pe.sections:
                section_name = section.Name.decode('utf-8', errors='ignore').strip('\x00')
                if section_name and section_name not in self.suspicious_sections:
                    unusual_sections += 1
            
            # Check 3: Small number of imports
            num_imports = 0
            if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                for entry in pe.DIRECTORY_ENTRY_IMPORT:
                    num_imports += len(entry.imports)
            
            # Check 4: Entry point in unusual section
            entry_point = pe.OPTIONAL_HEADER.AddressOfEntryPoint
            entry_section = None
            for section in pe.sections:
                if (section.VirtualAddress <= entry_point < 
                    section.VirtualAddress + section.Misc_VirtualSize):
                    entry_section = section.Name.decode('utf-8', errors='ignore').strip('\x00')
                    break
            
            unusual_entry = 0.0
            if entry_section and entry_section != '.text':
                unusual_entry = 1.0
            
            # Calculate heuristic score
            score = 0.0
            if high_entropy_sections >= 2:
                score += 0.3
            if unusual_sections >= 2:
                score += 0.2
            if num_imports < 10:
                score += 0.3
            if unusual_entry:
                score += 0.2
            
            features['packer_confidence'] = min(score, 1.0)
            if score > 0.5:
                features['is_packed'] = 1.0
            
            pe.close()
        
        except Exception as e:
            logger.warning(f"Error detecting packer: {e}")
        
        return features
    
    def analyze_strings(self, binary_path: str) -> Dict[str, float]:
        """
        Phân tích strings để detect obfuscation
        """
        features = {
            'string_entropy': 0.0,
            'avg_string_length': 0.0,
            'printable_ratio': 0.0,
            'base64_strings': 0,
            'hex_strings': 0,
            'suspicious_strings': 0,
            'string_diversity': 0.0
        }
        
        try:
            # Extract strings (min length 4)
            strings = self._extract_strings(binary_path, min_length=4)
            
            if not strings:
                return features
            
            # Calculate metrics
            all_strings = ''.join(strings)
            features['string_entropy'] = self.calculate_entropy(all_strings.encode('utf-8'))
            features['avg_string_length'] = np.mean([len(s) for s in strings])
            
            # Printable ratio
            printable = sum(1 for s in all_strings if s.isprintable())
            features['printable_ratio'] = printable / len(all_strings) if all_strings else 0
            
            # Detect encoded strings
            base64_pattern = re.compile(r'^[A-Za-z0-9+/]{20,}={0,2}$')
            hex_pattern = re.compile(r'^[0-9A-Fa-f]{16,}$')
            
            for s in strings:
                if base64_pattern.match(s):
                    features['base64_strings'] += 1
                if hex_pattern.match(s):
                    features['hex_strings'] += 1
            
            # Suspicious patterns
            suspicious_keywords = [
                'shellcode', 'payload', 'inject', 'hook', 'bypass',
                'decrypt', 'encode', 'obfuscate', 'hide'
            ]
            
            for s in strings:
                s_lower = s.lower()
                if any(keyword in s_lower for keyword in suspicious_keywords):
                    features['suspicious_strings'] += 1
            
            # String diversity (unique strings ratio)
            features['string_diversity'] = len(set(strings)) / len(strings) if strings else 0
        
        except Exception as e:
            logger.warning(f"Error analyzing strings: {e}")
        
        return features
    
    def _extract_strings(self, binary_path: str, min_length: int = 4) -> List[str]:
        """Extract ASCII strings from binary"""
        strings = []
        
        try:
            with open(binary_path, 'rb') as f:
                data = f.read()
            
            # Find ASCII strings
            current = []
            for byte in data:
                if 32 <= byte <= 126:  # Printable ASCII
                    current.append(chr(byte))
                else:
                    if len(current) >= min_length:
                        strings.append(''.join(current))
                    current = []
            
            if len(current) >= min_length:
                strings.append(''.join(current))
        
        except Exception as e:
            logger.warning(f"Error extracting strings: {e}")
        
        return strings
    
    def analyze_pe_structure(self, binary_path: str) -> Dict[str, float]:
        """
        Phân tích cấu trúc PE để detect anomalies
        """
        features = {
            'num_sections': 0,
            'suspicious_section_ratio': 0.0,
            'section_size_variance': 0.0,
            'has_tls': 0.0,
            'has_resources': 0.0,
            'overlay_size': 0.0,
            'entry_point_anomaly': 0.0
        }
        
        try:
            pe = pefile.PE(binary_path)
            
            # Section analysis
            features['num_sections'] = len(pe.sections)
            
            section_sizes = []
            suspicious_count = 0
            
            for section in pe.sections:
                section_name = section.Name.decode('utf-8', errors='ignore').strip('\x00')
                section_sizes.append(section.SizeOfRawData)
                
                # Check for suspicious sections
                if section_name and section_name not in self.suspicious_sections:
                    suspicious_count += 1
            
            features['suspicious_section_ratio'] = (
                suspicious_count / features['num_sections'] 
                if features['num_sections'] > 0 else 0
            )
            
            # Section size variance (high variance might indicate packing)
            if len(section_sizes) > 1:
                features['section_size_variance'] = float(np.var(section_sizes))
            
            # TLS callback (often used by packers)
            if hasattr(pe, 'DIRECTORY_ENTRY_TLS'):
                features['has_tls'] = 1.0
            
            # Resources
            if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
                features['has_resources'] = 1.0
            
            # Overlay (data after PE file)
            overlay_offset = pe.get_overlay_data_start_offset()
            if overlay_offset is not None:
                file_size = len(open(binary_path, 'rb').read())
                features['overlay_size'] = file_size - overlay_offset
            
            pe.close()
        
        except Exception as e:
            logger.warning(f"Error analyzing PE structure: {e}")
        
        return features
    
    def extract_all_features(self, binary_path: str) -> Dict[str, float]:
        """
        Trích xuất tất cả advanced features
        """
        all_features = {}
        
        # Entropy analysis
        entropy_features = self.analyze_section_entropy(binary_path)
        all_features.update(entropy_features)
        
        # Packer detection
        packer_features = self.detect_packer(binary_path)
        all_features.update(packer_features)
        
        # String analysis
        string_features = self.analyze_strings(binary_path)
        all_features.update(string_features)
        
        # PE structure
        pe_features = self.analyze_pe_structure(binary_path)
        all_features.update(pe_features)
        
        return all_features