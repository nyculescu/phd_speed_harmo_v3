# sumo_config.py
"""
SUMO Configuration Module

Handles loading, managing, and building SUMO command configurations
for the DRL-VSL traffic environment.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from copy import deepcopy

logger = logging.getLogger(__name__)

# Define available presets
PRESETS = {
    'default': 'default_sumo_config.yaml',
    'training': 'training_sumo_config.yaml', 
    'evaluation': 'evaluation_sumo_config.yaml',
    'debug': 'debug_sumo_config.yaml',
    'performance': 'performance_sumo_config.yaml',
    'high_fidelity': 'experiment_specific/high_fidelity.yaml'
}


class SumoConfig:
    """
    Manages SUMO configuration settings and command building.
    
    Supports loading from YAML files with inheritance, runtime updates,
    and preset configurations.
    """
    
    def __init__(self, config_dict: Dict[str, Any], base_path: Optional[Path] = None):
        """
        Initialize SUMO configuration.
        
        Args:
            config_dict: Configuration dictionary
            base_path: Base path for resolving relative paths in configs
        """
        self.config = deepcopy(config_dict)
        self.base_path = base_path or Path.cwd()
        self._validate_config()
        
    def _validate_config(self):
        """Validate configuration structure and set defaults."""
        # Ensure main sections exist
        if 'sumo' not in self.config:
            self.config['sumo'] = {}
        if 'files' not in self.config:
            self.config['files'] = {}
        if 'paths' not in self.config:
            self.config['paths'] = {}
            
        # Set default values for SUMO section
        sumo_defaults = {
            'binary': 'sumo-gui',
            'step_length': 1.0,
            'default_action_step_length': 0.2,
            'start': True,
            'default_emergencydecel': 7.0,
            'time_to_teleport': -1,
            'collision_action': 'warn',
            'random_depart_offset': 3600,
            'lateral_resolution': 0.2,
            'no_step_log': True,
            'no_warnings': True,
            'verbose': False,
            'additional_options': []
        }
        
        for key, default_value in sumo_defaults.items():
            if key not in self.config['sumo']:
                self.config['sumo'][key] = default_value
                
        # Set default file paths
        file_defaults = {
            'net_file': '../3_2_merge.net.xml',
            'additional_files': '../loops_detectors.add.xml',
            'gui_settings_file': '../colored.view.xml'
        }
        
        for key, default_value in file_defaults.items():
            if key not in self.config['files']:
                self.config['files'][key] = default_value
                
        # Set default paths
        path_defaults = {
            'sumo_home_env': 'SUMO_HOME',
            'generated_configs_dir': 'traffic_environment/sumo/generated_configs',
            'generated_flows_dir': 'traffic_environment/sumo/generated_flows',
            'sumo_logs_dir': 'logs/sumo_log'
        }
        
        for key, default_value in path_defaults.items():
            if key not in self.config['paths']:
                self.config['paths'][key] = default_value
    
    def get_sumo_binary(self, override: Optional[str] = None) -> str:
        """
        Get the SUMO binary path.
        
        Args:
            override: Optional override for the binary path
            
        Returns:
            Path to SUMO binary
        """
        if override:
            return override
            
        binary = self.config['sumo']['binary']
        
        # Check if it's a full path
        if os.path.isabs(binary) or os.path.exists(binary):
            return binary
            
        # Try to find it in SUMO_HOME
        sumo_home_env = self.config['paths'].get('sumo_home_env', 'SUMO_HOME')
        sumo_home = os.environ.get(sumo_home_env)
        
        if sumo_home:
            full_path = os.path.join(sumo_home, 'bin', binary)
            if os.path.exists(full_path):
                return full_path
                
        # Return as-is and hope it's in PATH
        return binary
    
    def get_sumo_cmd(self, 
                     port: int,
                     sim_length: int,
                     config_file: str,
                     log_file: Optional[str] = None,
                     binary_override: Optional[str] = None) -> List[str]:
        """
        Build the SUMO command with all parameters.
        
        Args:
            port: TraCI port number
            sim_length: Simulation length in seconds
            config_file: Path to SUMO config file
            log_file: Optional log file path
            binary_override: Optional binary path override
            
        Returns:
            List of command arguments
        """
        sumo_cfg = self.config['sumo']
        
        # Start with binary
        cmd = [self.get_sumo_binary(binary_override)]
        
        # Add config file
        cmd.extend(['-c', config_file])
        
        # Add standard options
        if sumo_cfg.get('start', True):
            cmd.append('--start')
            
        cmd.extend([
            '--default.emergencydecel', str(sumo_cfg['default_emergencydecel']),
            '--random-depart-offset', str(sumo_cfg['random_depart_offset']),
            '--remote-port', str(port),
            '--step-length', str(sumo_cfg['step_length']),
            '--default.action-step-length', str(sumo_cfg['default_action_step_length']),
            '--end', str(sim_length),
            '--time-to-teleport', str(sumo_cfg['time_to_teleport']),
            '--collision.action', sumo_cfg['collision_action'],
            '--lateral-resolution', str(sumo_cfg.get('lateral_resolution', 0.2))
        ])
        
        # Add boolean flags
        if sumo_cfg.get('no_step_log', True):
            cmd.append('--no-step-log')
        if sumo_cfg.get('no_warnings', True):
            cmd.append('--no-warnings')
        if sumo_cfg.get('verbose', False):
            cmd.append('--verbose')
            
        # Add log file if provided
        if log_file:
            cmd.extend(['--log', log_file])
            
        # Add any additional options
        additional_options = sumo_cfg.get('additional_options', [])
        if additional_options:
            cmd.extend(additional_options)
            
        return cmd
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration values at runtime.
        
        Args:
            updates: Dictionary of updates (can be nested)
        """
        def deep_update(base_dict: dict, update_dict: dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
                    
        deep_update(self.config, updates)
        self._validate_config()
    
    def get_config_value(self, path: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-separated path.
        
        Args:
            path: Dot-separated path (e.g., 'sumo.step_length')
            default: Default value if path not found
            
        Returns:
            Configuration value or default
        """
        parts = path.split('.')
        value = self.config
        
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
                
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return deepcopy(self.config)


def load_yaml_with_inheritance(file_path: Path, visited: Optional[set] = None) -> Dict[str, Any]:
    """
    Load YAML file with support for 'extends' inheritance.
    
    Args:
        file_path: Path to YAML file
        visited: Set of visited files to prevent circular imports
        
    Returns:
        Merged configuration dictionary
    """
    if visited is None:
        visited = set()
        
    # Prevent circular imports
    abs_path = file_path.absolute()
    if abs_path in visited:
        raise ValueError(f"Circular inheritance detected: {abs_path}")
    visited.add(abs_path)
    
    # Load the YAML file
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f) or {}
    
    # Check for inheritance
    if 'extends' in config:
        extends_path = config.pop('extends')
        
        # Resolve relative paths
        if not os.path.isabs(extends_path):
            base_dir = file_path.parent
            extends_path = base_dir / extends_path
        else:
            extends_path = Path(extends_path)
            
        # Load parent config
        parent_config = load_yaml_with_inheritance(extends_path, visited)
        
        # Deep merge with parent (child overrides parent)
        def deep_merge(base: dict, override: dict) -> dict:
            result = deepcopy(base)
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = deepcopy(value)
            return result
            
        config = deep_merge(parent_config, config)
    
    return config


def load_sumo_config(config_path: Union[str, Path]) -> SumoConfig:
    """
    Load SUMO configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        SumoConfig instance
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading SUMO configuration from: {config_path}")
    
    # Load with inheritance support
    config_dict = load_yaml_with_inheritance(config_path)
    
    return SumoConfig(config_dict, config_path.parent)


def get_default_config_dir() -> Path:
    """Get the default configuration directory."""
    return Path("traffic_environment/sumo/sumo_configs")


def get_default_sumo_config() -> SumoConfig:
    """
    Get the default SUMO configuration.
    
    Returns:
        Default SumoConfig instance
    """
    config_dir = get_default_config_dir()
    default_config_path = config_dir / "default_sumo_config.yaml"
    
    if default_config_path.exists():
        return load_sumo_config(default_config_path)
    else:
        logger.warning(f"Default config file not found at {default_config_path}, using built-in defaults")
        # Return built-in defaults
        default_config = {
            'sumo': {
                'binary': 'sumo-gui',
                'step_length': 1.0,
                'default_action_step_length': 0.2,
                'start': True,
                'default_emergencydecel': 7.0,
                'time_to_teleport': -1,
                'collision_action': 'warn',
                'random_depart_offset': 3600,
                'lateral_resolution': 0.2,
                'no_step_log': True,
                'no_warnings': True,
                'verbose': False,
                'additional_options': []
            },
            'files': {
                'net_file': '../3_2_merge.net.xml',
                'additional_files': '../loops_detectors.add.xml',
                'gui_settings_file': '../colored.view.xml'
            },
            'paths': {
                'sumo_home_env': 'SUMO_HOME',
                'generated_configs_dir': 'traffic_environment/sumo/generated_configs',
                'generated_flows_dir': 'traffic_environment/sumo/generated_flows',
                'sumo_logs_dir': 'logs/sumo_log'
            }
        }
        return SumoConfig(default_config)


def get_preset_config(preset_name: str) -> SumoConfig:
    """
    Get a preset SUMO configuration.
    
    Args:
        preset_name: Name of the preset (e.g., 'training', 'evaluation')
        
    Returns:
        SumoConfig instance for the preset
        
    Raises:
        ValueError: If preset name is not recognized
    """
    if preset_name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
    
    config_dir = get_default_config_dir()
    preset_file = PRESETS[preset_name]
    config_path = config_dir / preset_file
    
    if not config_path.exists():
        raise FileNotFoundError(f"Preset configuration file not found: {config_path}")
    
    logger.info(f"Loading preset configuration: {preset_name}")
    return load_sumo_config(config_path)


def create_sumo_config_from_dict(config_dict: Dict[str, Any]) -> SumoConfig:
    """
    Create a SumoConfig instance from a dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        SumoConfig instance
    """
    return SumoConfig(config_dict)


# Utility functions for common operations

def list_available_presets() -> List[str]:
    """List all available preset names."""
    return list(PRESETS.keys())


def validate_sumo_binary(binary_path: str) -> bool:
    """
    Validate that a SUMO binary exists and is executable.
    
    Args:
        binary_path: Path to SUMO binary
        
    Returns:
        True if valid, False otherwise
    """
    if os.path.exists(binary_path):
        return os.access(binary_path, os.X_OK)
    
    # Check if it's in PATH
    from shutil import which
    return which(binary_path) is not None


def merge_configs(*configs: Union[Dict[str, Any], SumoConfig]) -> SumoConfig:
    """
    Merge multiple configurations, with later ones overriding earlier ones.
    
    Args:
        *configs: Configuration dictionaries or SumoConfig instances
        
    Returns:
        Merged SumoConfig instance
    """
    result = {}
    
    for config in configs:
        if isinstance(config, SumoConfig):
            config_dict = config.to_dict()
        else:
            config_dict = config
            
        def deep_merge(base: dict, override: dict):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = deepcopy(value)
                    
        deep_merge(result, config_dict)
    
    return SumoConfig(result)


# Example usage and testing
if __name__ == "__main__":
    # Test loading default config
    print("Testing SUMO configuration module...")
    
    try:
        # Get default config
        default_config = get_default_sumo_config()
        print(f"✓ Loaded default config: step_length = {default_config.config['sumo']['step_length']}")
        
        # Get preset configs
        for preset in ['training', 'evaluation', 'performance']:
            try:
                preset_config = get_preset_config(preset)
                print(f"✓ Loaded {preset} preset: binary = {preset_config.config['sumo']['binary']}")
            except FileNotFoundError:
                print(f"✗ {preset} preset file not found (run setup_sumo_config.py first)")
        
        # Test command building
        cmd = default_config.get_sumo_cmd(
            port=8000,
            sim_length=3600,
            config_file="test.sumocfg",
            log_file="test.log"
        )
        print(f"✓ Built SUMO command with {len(cmd)} arguments")
        
        # Test config updates
        default_config.update_config({
            'sumo': {'step_length': 2.0, 'binary': 'sumo'}
        })
        print(f"✓ Updated config: step_length = {default_config.config['sumo']['step_length']}")
        
        print("\nAll tests passed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")