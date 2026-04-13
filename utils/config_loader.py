"""
Configuration loader for the proctoring system.
Loads and validates YAML configuration files.
"""

import yaml
import os
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """Loads and manages configuration from YAML files."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file."""
        # Get absolute path
        if not os.path.isabs(self.config_path):
            # Try relative to current directory
            base_dir = Path(__file__).parent.parent
            config_file = base_dir / self.config_path
        else:
            config_file = Path(self.config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Ensure required directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create required directories if they don't exist."""
        dirs_to_create = [
            self.config.get('video', {}).get('recording_path', './recordings'),
            self.config.get('logging', {}).get('log_path', './logs'),
            self.config.get('global', {}).get('output_path', './reports'),
            self.config.get('reporting', {}).get('output_dir', './reports/generated'),
            self.config.get('reporting', {}).get('image_dir', './reports/generated/images'),
        ]
        
        for dir_path in dirs_to_create:
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key (supports dot notation).
        
        Args:
            key: Configuration key (e.g., 'detection.face.min_confidence')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if not self.config:
            return default
        
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self.config
    
    def reload(self):
        """Reload configuration from file."""
        self._load_config()


# Global config instance
_config_instance = None


def get_config(config_path: str = "config/config.yaml") -> ConfigLoader:
    """
    Get or create global configuration instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigLoader instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)
    return _config_instance

