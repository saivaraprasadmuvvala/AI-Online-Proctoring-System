"""
Violation logger for structured violation logging.
Logs violations with timestamps and metadata.
"""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, List


class ViolationLogger:
    """Logs violations with structured data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize violation logger.
        
        Args:
            config: Configuration dictionary (optional)
        """
        if config:
            output_path = config.get('global', {}).get('output_path', './reports')
        else:
            output_path = './reports'
        
        self.log_file = os.path.join(output_path, "violations.json")
        self.violations: List[Dict[str, Any]] = []
        
        # Load existing violations if file exists
        self._load_violations()
    
    def _load_violations(self):
        """Load existing violations from file."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    self.violations = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.violations = []
    
    def log_violation(self, violation_type: str, timestamp: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Log a violation with timestamp and metadata.
        
        Args:
            violation_type: Type of violation
            timestamp: Timestamp string (defaults to current time)
            metadata: Optional metadata dictionary
        """
        entry = {
            'type': violation_type,
            'timestamp': timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
            'metadata': metadata or {}
        }
        self.violations.append(entry)
        self._save_to_file()
    
    def _save_to_file(self):
        """Save violations to JSON file."""
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump(self.violations, f, indent=2)
    
    def get_violations(self) -> List[Dict[str, Any]]:
        """
        Get all logged violations.
        
        Returns:
            List of violation dictionaries
        """
        return self.violations
    
    def get_violations_by_type(self, violation_type: str) -> List[Dict[str, Any]]:
        """
        Get violations filtered by type.
        
        Args:
            violation_type: Type of violation to filter
            
        Returns:
            List of matching violation dictionaries
        """
        return [v for v in self.violations if v['type'] == violation_type]
    
    def clear_violations(self):
        """Clear all violations."""
        self.violations = []
        self._save_to_file()

