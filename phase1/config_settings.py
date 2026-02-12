"""
Configuration management for valve leak detection
Removes hard-coded values from business logic
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class DetectionThresholds:
    """Leak detection amplitude thresholds."""
    severe: float = 5.0      # >5.0G = SEVERE LEAK (90-100%)
    moderate: float = 3.5    # 3.5-5.0G = MODERATE (70-90%)
    likely: float = 3.0      # 3.0-4.0G = LIKELY (60-80%)
    possible: float = 2.0    # 2.0-3.0G = POSSIBLE (40-60%)
    normal: float = 2.0      # <2.0G = NORMAL (0-30%)

@dataclass
class WRPMConfig:
    """WRPM parser configuration."""
    # Default full-scale values for calibration
    full_scale_g: float = 10.0      # Default for AE sensors (G)
    full_scale_psi: float = 2000.0  # Default for pressure sensors (PSI)
    
    # AE sensor file extensions (priority order)
    ae_extensions: tuple = ('.SDD', '.S&&', '.S$$')
    
    # Machine type detection patterns
    compressor_patterns: tuple = ('C', 'Compressor', 'Unit')
    engine_patterns: tuple = ('E', 'Engine')

@dataclass
class ProcessingConfig:
    """Data processing configuration."""
    batch_size: int = 10
    default_sampling_points: int = 355  # Standard for compressors
    crank_angle_range: int = 360        # Compressors use 0-360°
    engine_crank_angle_range: int = 720 # Engines use 0-720°

@dataclass
class OutputConfig:
    """Output formatting configuration."""
    timestamp_format: str = '%Y-%m-%d %H:%M:%S'
    csv_float_precision: int = 3
    json_indent: int = 2

class Config:
    """Centralized configuration with dot notation access."""
    
    def __init__(self, config_file: Optional[str] = None):
        # Initialize all configuration sections
        self.detection = DetectionThresholds()
        self.wrpm = WRPMConfig()
        self.processing = ProcessingConfig()
        self.output = OutputConfig()
        
        # Color coding for UI (from app.py)
        self.colors = {
            'leak_high': {'bg': '#ffebee', 'text': '#c62828'},    # Red
            'leak_medium': {'bg': '#fff3e0', 'text': '#f57c00'},  # Orange
            'normal': {'bg': '#e8f5e9', 'text': '#2e7d32'},       # Green
            'waveform': {
                'normal': '#2e7d32',      # Green
                'leak': '#c62828',        # Red
                'fill': 'rgba(66, 165, 245, 0.4)',  # Blue fill
                'threshold': '#ff9800'    # Orange threshold line
            }
        }
        
        # Load from config file if provided
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update detection thresholds
            if 'detection' in config_data:
                for key, value in config_data['detection'].items():
                    if hasattr(self.detection, key):
                        setattr(self.detection, key, value)
            
            # Update WRPM config
            if 'wrpm' in config_data:
                for key, value in config_data['wrpm'].items():
                    if hasattr(self.wrpm, key):
                        setattr(self.wrpm, key, value)
            
            print(f"✅ Loaded configuration from {config_file}")
            
        except Exception as e:
            print(f"⚠️  Could not load config file: {e}")
            print("💡 Using default configuration")
    
    def save_config(self, config_file: str) -> None:
        """Save current configuration to JSON file."""
        try:
            config_data = {
                'detection': self.detection.__dict__,
                'wrpm': self.wrpm.__dict__,
                'processing': self.processing.__dict__,
                'output': self.output.__dict__
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            print(f"✅ Saved configuration to {config_file}")
            
        except Exception as e:
            print(f"❌ Could not save config file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        parts = key.split('.')
        current = self
        
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        
        return current
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'detection': self.detection.__dict__,
            'wrpm': self.wrpm.__dict__,
            'processing': self.processing.__dict__,
            'output': self.output.__dict__,
            'colors': self.colors
        }
    
    def update_from_dict(self, config_dict: Dict) -> None:
        """Update configuration from dictionary."""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                if hasattr(section_obj, '__dict__'):
                    for key, value in values.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
                elif isinstance(section_obj, dict):
                    section_obj.update(values)

# Global instance
config = Config()

# Helper functions for common configuration needs
def get_leak_threshold(amplitude: float) -> tuple[str, float]:
    """Get leak classification based on amplitude."""
    if amplitude >= config.detection.severe:
        return ("SEVERE", 0.95)
    elif amplitude >= config.detection.moderate:
        return ("MODERATE", 0.80)
    elif amplitude >= config.detection.likely:
        return ("LIKELY", 0.70)
    elif amplitude >= config.detection.possible:
        return ("POSSIBLE", 0.50)
    else:
        return ("NORMAL", 0.20)

def get_color_for_probability(probability: float) -> tuple[str, str]:
    """Get background and text colors for leak probability."""
    if probability >= 0.5:  # 50% or higher
        return config.colors['leak_high']['bg'], config.colors['leak_high']['text']
    elif probability >= 0.3:  # 30-50%
        return config.colors['leak_medium']['bg'], config.colors['leak_medium']['text']
    else:  # Below 30%
        return config.colors['normal']['bg'], config.colors['normal']['text']

# Quick test when run directly
if __name__ == "__main__":
    print("🧪 Testing configuration system...")
    print("=" * 50)
    
    print("\n📊 Current Configuration:")
    print(f"  Detection thresholds:")
    print(f"    Severe: {config.detection.severe}G")
    print(f"    Moderate: {config.detection.moderate}G")
    print(f"    Likely: {config.detection.likely}G")
    print(f"    Possible: {config.detection.possible}G")
    print(f"    Normal: <{config.detection.normal}G")
    
    print(f"\n  WRPM Settings:")
    print(f"    Full-scale G: {config.wrpm.full_scale_g}")
    print(f"    AE extensions: {config.wrpm.ae_extensions}")
    
    print(f"\n  Dot notation access:")
    print(f"    config.get('detection.severe'): {config.get('detection.severe')}")
    print(f"    config.get('colors.waveform.normal'): {config.get('colors.waveform.normal')}")
    
    print(f"\n  Helper functions:")
    test_amplitudes = [1.5, 2.5, 3.5, 5.5]
    for amp in test_amplitudes:
        classification, confidence = get_leak_threshold(amp)
        print(f"    {amp}G → {classification} ({confidence:.0%} confidence)")
    
    print("\n✅ Configuration system ready!")