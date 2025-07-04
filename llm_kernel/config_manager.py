"""
Configuration Management System

This module handles hierarchical configuration management for the LLM kernel,
supporting user, project, and notebook-level settings.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Manages hierarchical configuration for the LLM kernel."""
    
    def __init__(self):
        self.default_config = {
            # Model settings
            'default_model': 'gpt-4o',
            'fallback_model': 'gpt-4o-mini',
            'model_temperature': 0.7,
            'model_max_tokens': None,
            
            # Context management
            'context_strategy': 'smart',
            'max_context_tokens': 4000,
            'max_context_cells': 20,
            'auto_prune': True,
            'pruning_strategy': 'hybrid',
            'pruning_threshold': 0.7,
            
            # Execution tracking
            'track_dependencies': True,
            'track_variables': True,
            'track_imports': True,
            
            # Display settings
            'show_token_usage': True,
            'show_model_info': True,
            'rich_output': True,
            'comparison_tabs': True,
            
            # Performance settings
            'parallel_queries': True,
            'max_parallel_workers': 4,
            'request_timeout': 30,
            
            # Logging
            'log_level': 'INFO',
            'log_exchanges': False,
            'log_context_changes': False,
        }
        
        self.user_config_dir = Path.home() / '.llm-kernel'
        self.user_config_file = self.user_config_dir / 'config.json'
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from all sources in priority order."""
        config = self.default_config.copy()
        
        # 1. Load user global settings
        user_config = self.load_user_config()
        if user_config:
            config.update(user_config)
        
        # 2. Load project settings
        project_config = self.load_project_config()
        if project_config:
            config.update(project_config)
        
        # 3. Load notebook metadata (handled by kernel)
        # This will be loaded separately by the kernel
        
        # 4. Environment variable overrides
        env_config = self.load_env_overrides()
        if env_config:
            config.update(env_config)
        
        return config
    
    def load_user_config(self) -> Optional[Dict[str, Any]]:
        """Load user-level configuration."""
        if not self.user_config_file.exists():
            return None
        
        try:
            with open(self.user_config_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load user config: {e}")
            return None
    
    def load_project_config(self) -> Optional[Dict[str, Any]]:
        """Load project-level configuration."""
        # Look for .llm-kernel.json in current directory and parents
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            config_file = parent / '.llm-kernel.json'
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        return json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    print(f"Warning: Could not load project config from {config_file}: {e}")
                    return None
        
        return None
    
    def load_env_overrides(self) -> Dict[str, Any]:
        """Load configuration overrides from environment variables."""
        overrides = {}
        
        # Map environment variables to config keys
        env_mappings = {
            'LLM_KERNEL_DEFAULT_MODEL': 'default_model',
            'LLM_KERNEL_CONTEXT_STRATEGY': 'context_strategy',
            'LLM_KERNEL_MAX_TOKENS': ('max_context_tokens', int),
            'LLM_KERNEL_MAX_CELLS': ('max_context_cells', int),
            'LLM_KERNEL_AUTO_PRUNE': ('auto_prune', bool),
            'LLM_KERNEL_PRUNING_STRATEGY': 'pruning_strategy',
            'LLM_KERNEL_PRUNING_THRESHOLD': ('pruning_threshold', float),
            'LLM_KERNEL_LOG_LEVEL': 'log_level',
            'LLM_KERNEL_PARALLEL': ('parallel_queries', bool),
            'LLM_KERNEL_TIMEOUT': ('request_timeout', int),
        }
        
        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if isinstance(config_key, tuple):
                    key, type_func = config_key
                    try:
                        if type_func == bool:
                            overrides[key] = value.lower() in ('true', '1', 'yes', 'on')
                        else:
                            overrides[key] = type_func(value)
                    except ValueError:
                        print(f"Warning: Invalid value for {env_var}: {value}")
                else:
                    overrides[config_key] = value
        
        return overrides
    
    def save_user_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to user-level file."""
        try:
            # Create directory if it doesn't exist
            self.user_config_dir.mkdir(exist_ok=True)
            
            with open(self.user_config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            return True
        except IOError as e:
            print(f"Error saving user config: {e}")
            return False
    
    def save_project_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to project-level file."""
        config_file = Path.cwd() / '.llm-kernel.json'
        
        try:
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            return True
        except IOError as e:
            print(f"Error saving project config: {e}")
            return False
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get information about configuration sources."""
        info = {
            'user_config_file': str(self.user_config_file),
            'user_config_exists': self.user_config_file.exists(),
            'project_config_file': None,
            'project_config_exists': False,
        }
        
        # Find project config file
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            config_file = parent / '.llm-kernel.json'
            if config_file.exists():
                info['project_config_file'] = str(config_file)
                info['project_config_exists'] = True
                break
        
        return info
    
    def create_example_configs(self):
        """Create example configuration files."""
        # User config example
        user_example = {
            "default_model": "gpt-4o-mini",
            "context_strategy": "smart",
            "max_context_tokens": 4000,
            "auto_prune": True,
            "show_token_usage": True,
            "log_level": "INFO"
        }
        
        # Project config example
        project_example = {
            "default_model": "claude-3-sonnet",
            "context_strategy": "dependency",
            "max_context_tokens": 6000,
            "pruning_strategy": "hybrid",
            "track_dependencies": True,
            "project_specific_setting": "example_value"
        }
        
        # Create user config directory and example
        self.user_config_dir.mkdir(exist_ok=True)
        
        user_example_file = self.user_config_dir / 'config.example.json'
        with open(user_example_file, 'w') as f:
            json.dump(user_example, f, indent=2)
        
        # Create project example in current directory
        project_example_file = Path.cwd() / '.llm-kernel.example.json'
        with open(project_example_file, 'w') as f:
            json.dump(project_example, f, indent=2)
        
        print(f"Created example configs:")
        print(f"  User: {user_example_file}")
        print(f"  Project: {project_example_file}")
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize configuration values."""
        validated = {}
        
        # Validate each setting
        validators = {
            'context_strategy': lambda x: x if x in ['chronological', 'dependency', 'smart', 'manual'] else 'smart',
            'pruning_strategy': lambda x: x if x in ['semantic', 'recency', 'dependency', 'hybrid'] else 'hybrid',
            'max_context_tokens': lambda x: max(100, min(32000, int(x))) if isinstance(x, (int, str)) else 4000,
            'max_context_cells': lambda x: max(1, min(100, int(x))) if isinstance(x, (int, str)) else 20,
            'pruning_threshold': lambda x: max(0.0, min(1.0, float(x))) if isinstance(x, (int, float, str)) else 0.7,
            'model_temperature': lambda x: max(0.0, min(2.0, float(x))) if isinstance(x, (int, float, str)) else 0.7,
            'request_timeout': lambda x: max(5, min(300, int(x))) if isinstance(x, (int, str)) else 30,
            'max_parallel_workers': lambda x: max(1, min(10, int(x))) if isinstance(x, (int, str)) else 4,
            'log_level': lambda x: x.upper() if x.upper() in ['DEBUG', 'INFO', 'WARNING', 'ERROR'] else 'INFO',
        }
        
        for key, value in config.items():
            if key in validators:
                try:
                    validated[key] = validators[key](value)
                except (ValueError, TypeError):
                    # Use default value if validation fails
                    validated[key] = self.default_config.get(key, value)
            else:
                validated[key] = value
        
        return validated
    
    def merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        merged = base.copy()
        merged.update(override)
        return merged
    
    def get_config_diff(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """Get differences between two configurations."""
        diff = {}
        
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            val1 = config1.get(key)
            val2 = config2.get(key)
            
            if val1 != val2:
                diff[key] = {
                    'old': val1,
                    'new': val2
                }
        
        return diff
