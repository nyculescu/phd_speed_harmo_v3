# drl_vsl_integration.py
"""
Integration utilities for the modular SAR framework.

This module provides helper functions and utilities for using the SAR framework.
"""

import os
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
import json

from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit

logger = logging.getLogger(__name__)

# Global SAR configuration (can be loaded from file)
DEFAULT_SAR_CONFIG = {
    'max_flow': 10000.0,
    'max_occupancy': 100.0,
    'max_queue_length': 575.0 * 3 / 7
}


def load_sar_config_from_file(filepath: str) -> Dict[str, Any]:
    """
    Load SAR configuration from a JSON file.
    
    Args:
        filepath: Path to the configuration file
        
    Returns:
        Dictionary with SAR configuration
    """
    sar_config = DEFAULT_SAR_CONFIG.copy()
    
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Update with loaded values
            if "bounds" in data:
                sar_config.update(data["bounds"])
            else:
                sar_config.update(data)
            
            logger.info(f"Loaded SAR configuration from {filepath}")
        except Exception as e:
            logger.warning(f"Could not load config from {filepath}: {e}, using defaults")
    
    return sar_config


def get_model_config_path(model_name: str, config_dir: str = "rl_models/optuna_params") -> Optional[Path]:
    """
    Get the configuration file path for a given model.
    
    Args:
        model_name: Name of the model
        config_dir: Directory containing configuration files
        
    Returns:
        Path to config file if it exists, None otherwise
    """
    config_path = Path(config_dir) / f"best_optuna_hyperparams_{model_name}.json"
    return config_path if config_path.exists() else None


def merge_sar_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple SAR configurations, with later ones overriding earlier ones.
    
    Args:
        *configs: Variable number of configuration dictionaries
        
    Returns:
        Merged configuration dictionary
    """
    result = {}
    for config in configs:
        result.update(config)
    return result


# Preset SAR configurations for common scenarios
PRESET_SAR_CONFIGS = {
    'default': {
        'max_flow': 10000.0,
        'max_occupancy': 100.0,
        'max_queue_length': 575.0 * 3 / 7
    },
    'highway': {
        'max_flow': 12000.0,
        'max_occupancy': 100.0,
        'max_queue_length': 500.0
    },
    'urban': {
        'max_flow': 8000.0,
        'max_occupancy': 100.0,
        'max_queue_length': 300.0
    },
    'congested': {
        'max_flow': 6000.0,
        'max_occupancy': 100.0,
        'max_queue_length': 600.0
    }
}


def get_preset_sar_config(preset_name: str) -> Dict[str, Any]:
    """
    Get a preset SAR configuration.
    
    Args:
        preset_name: Name of the preset ('default', 'highway', 'urban', 'congested')
        
    Returns:
        SAR configuration dictionary
        
    Raises:
        ValueError: If preset name is not recognized
    """
    if preset_name not in PRESET_SAR_CONFIGS:
        available = ', '.join(PRESET_SAR_CONFIGS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    return PRESET_SAR_CONFIGS[preset_name].copy()


def create_traffic_env_from_config(port: int,
                                   model_name: str,
                                   model_idx: int,
                                   sim_length: int,
                                   base_gen_car_distrib: list,
                                   num_of_episodes: int,
                                   state_representation: Union[str, Any],
                                   action_strategy: Union[str, Any],
                                   reward_function: Union[str, Any],
                                   vsl_enforcement: str = "recommend",
                                   sumo_binary_path_override: Optional[str] = None,
                                   sar_config: Optional[Dict[str, Any]] = None,
                                   sumo_config: Optional[Any] = None,
                                   sumo_preset: Optional[str] = None,
                                   wrap_with_monitor: bool = True,
                                   time_limit: Optional[int] = None) -> Union[Any, Monitor]:
    """
    Create a TrafficEnv with the given configuration.
    
    This is a helper function that creates a TrafficEnv instance with proper
    initialization of SAR components.
    
    Args:
        port: SUMO port
        model_name: Model name
        model_idx: Model index
        sim_length: Simulation length in seconds
        base_gen_car_distrib: Base car distribution
        num_of_episodes: Number of episodes
        state_representation: State representation (string or object)
        action_strategy: Action strategy (string or object)
        reward_function: Reward function (string or object)
        vsl_enforcement: VSL enforcement mode
        sumo_binary_path_override: Optional SUMO binary path
        sar_config: SAR configuration dict
        sumo_config: SUMO configuration
        sumo_preset: SUMO preset name
        wrap_with_monitor: Whether to wrap with Monitor
        time_limit: Optional time limit in steps
        
    Returns:
        TrafficEnv or Monitor-wrapped TrafficEnv
    """
    from .drl_vsl import TrafficEnv, create_sumocfg
    from .sar_framework import (
        create_state_representation,
        create_action_strategy,
        create_reward_function
    )
    
    # Load SAR configuration
    if sar_config is None:
        sar_config = DEFAULT_SAR_CONFIG.copy()
    
    # Convert string specifications to objects if needed
    if isinstance(state_representation, str):
        state_repr_obj = create_state_representation(state_representation, sar_config)
    else:
        state_repr_obj = state_representation
        
    if isinstance(action_strategy, str):
        action_strat_obj = create_action_strategy(action_strategy, sar_config)
    else:
        action_strat_obj = action_strategy
        
    if isinstance(reward_function, str):
        reward_func_obj = create_reward_function(reward_function, sar_config)
    else:
        reward_func_obj = reward_function
    
    # Create SUMO config file
    effective_model_name = f"{model_name}_{model_idx}"
    create_sumocfg(effective_model_name)
    
    # Create environment
    env = TrafficEnv(
        port=port,
        model_name=model_name,
        model_idx=model_idx,
        sim_length=sim_length,
        base_gen_car_distrib=base_gen_car_distrib,
        num_of_episodes=num_of_episodes,
        state_representation=state_repr_obj,
        action_strategy=action_strat_obj,
        reward_function=reward_func_obj,
        vsl_enforcement=vsl_enforcement,
        sumo_binary_path_override=sumo_binary_path_override,
        sar_config=sar_config,
        sumo_config=sumo_config,
        sumo_preset=sumo_preset
    )
    
    # Apply time limit if specified
    if time_limit is not None:
        env = TimeLimit(env, max_episode_steps=time_limit)
    
    # Wrap with Monitor if requested
    if wrap_with_monitor:
        return Monitor(env)
    
    return env


def create_train_env_helper(idx: int,
                            model_name: str,
                            sim_length: int,
                            num_of_episodes: int,
                            reward_fn: str,
                            vsl_enforcement: str = "recommend",
                            state_representation: str = "full_metrics",
                            action_strategy: str = "absolute_speed",
                            sumo_port: Optional[int] = None,
                            sumo_binary: Optional[str] = None,
                            normalization_bounds_path: Optional[str] = None,
                            sumo_config: Optional[Any] = None,
                            sumo_preset: Optional[str] = None) -> Monitor:
    """
    Helper function to create a training environment.
    
    Args:
        idx: Environment index
        model_name: Model name
        sim_length: Simulation length
        num_of_episodes: Number of episodes
        reward_fn: Reward function name
        vsl_enforcement: VSL enforcement mode
        state_representation: State representation name
        action_strategy: Action strategy name
        sumo_port: SUMO port (default: 8000 + idx)
        sumo_binary: SUMO binary path
        normalization_bounds_path: Path to normalization bounds
        sumo_config: SUMO configuration
        sumo_preset: SUMO preset name
        
    Returns:
        Monitor-wrapped training environment
    """
    port = sumo_port if sumo_port is not None else 8000 + idx
    
    # Auto-detect normalization bounds path if not provided
    if normalization_bounds_path is None:
        bounds_path = Path("rl_models/optuna_params") / f"best_optuna_hyperparams_{model_name}.json"
        if bounds_path.exists():
            normalization_bounds_path = str(bounds_path)
    
    # Load SAR config
    sar_config = DEFAULT_SAR_CONFIG.copy()
    if normalization_bounds_path:
        sar_config = load_sar_config_from_file(normalization_bounds_path)
    
    return create_traffic_env_from_config(
        port=port,
        model_name=model_name,
        model_idx=idx,
        sim_length=sim_length,
        base_gen_car_distrib=["uniform", 2000],  # Default, will be overridden in reset
        num_of_episodes=num_of_episodes,
        state_representation=state_representation,
        action_strategy=action_strategy,
        reward_function=reward_fn,
        vsl_enforcement=vsl_enforcement,
        sumo_binary_path_override=sumo_binary,
        sar_config=sar_config,
        sumo_config=sumo_config,
        sumo_preset=sumo_preset,
        wrap_with_monitor=True
    )


def create_eval_env_helper(model_name: str,
                           sim_length: int,
                           reward_fn: str,
                           vsl_enforcement: str = "recommend",
                           state_representation: str = "full_metrics",
                           action_strategy: str = "absolute_speed",
                           sumo_port: Optional[int] = None,
                           sumo_binary: Optional[str] = None,
                           normalization_bounds_path: Optional[str] = None,
                           sumo_config: Optional[Any] = None,
                           sumo_preset: Optional[str] = None,
                           eval_idx: int = 0) -> Monitor:
    """
    Helper function to create an evaluation environment.
    
    Args:
        model_name: Model name
        sim_length: Simulation length
        reward_fn: Reward function name
        vsl_enforcement: VSL enforcement mode
        state_representation: State representation name
        action_strategy: Action strategy name
        sumo_port: SUMO port (default: 10000 + eval_idx)
        sumo_binary: SUMO binary path
        normalization_bounds_path: Path to normalization bounds
        sumo_config: SUMO configuration
        sumo_preset: SUMO preset name
        eval_idx: Evaluation environment index
        
    Returns:
        Monitor-wrapped evaluation environment with time limit
    """
    port = sumo_port if sumo_port is not None else 10000 + eval_idx
    
    # Auto-detect normalization bounds path if not provided
    if normalization_bounds_path is None:
        bounds_path = Path("rl_models/optuna_params") / f"best_optuna_hyperparams_{model_name}.json"
        if bounds_path.exists():
            normalization_bounds_path = str(bounds_path)
    
    # Load SAR config
    sar_config = DEFAULT_SAR_CONFIG.copy()
    if normalization_bounds_path:
        sar_config = load_sar_config_from_file(normalization_bounds_path)
    
    # Calculate time limit based on sim length
    max_episode_steps = sim_length // 60  # Assuming 60s aggregation
    
    return create_traffic_env_from_config(
        port=port,
        model_name=model_name,
        model_idx=eval_idx,
        sim_length=sim_length,
        base_gen_car_distrib=["uniform", 3000],  # Eval default
        num_of_episodes=1,
        state_representation=state_representation,
        action_strategy=action_strategy,
        reward_function=reward_fn,
        vsl_enforcement=vsl_enforcement,
        sumo_binary_path_override=sumo_binary,
        sar_config=sar_config,
        sumo_config=sumo_config,
        sumo_preset=sumo_preset,
        wrap_with_monitor=True,
        time_limit=max_episode_steps
    )


# Example usage and validation
if __name__ == "__main__":
    # Test configuration loading
    print("Testing SAR configuration utilities...")
    
    # Test preset configs
    for preset in PRESET_SAR_CONFIGS:
        config = get_preset_sar_config(preset)
        print(f"✓ Loaded {preset} preset: max_flow = {config['max_flow']}")
    
    # Test config merging
    base_config = get_preset_sar_config('default')
    custom_config = {'max_flow': 15000.0, 'custom_param': 42}
    merged = merge_sar_configs(base_config, custom_config)
    print(f"✓ Merged configs: max_flow = {merged['max_flow']}, custom_param = {merged.get('custom_param')}")
    
    # Example: Creating environments using helper functions
    from stable_baselines3.common.vec_env import SubprocVecEnv
    
    print("\nExample: Creating training environments...")
    
    # Create multiple training environments
    envs = SubprocVecEnv([
        lambda i=i: create_train_env_helper(
            i, "DQN_test", 3600, 10, "balanced", "recommend",
            "full_metrics", "absolute_speed"
        )
        for i in range(4)
    ])
    
    print("✓ Created 4 parallel training environments")
    
    # Create evaluation environment
    eval_env = create_eval_env_helper(
        "DQN_test", 3600, "balanced", "recommend",
        "full_metrics", "absolute_speed"
    )
    
    print("✓ Created evaluation environment")
    
    # Clean up
    envs.close()
    eval_env.close()
    
    print("\nAll tests passed!")