# drl_vsl_integration.py
"""
Integration wrapper to use the modular SAR framework with minimal changes
to existing training and evaluation code.

This module provides backward-compatible functions that allow gradual migration.
"""

import os
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging

from .drl_vsl import TrafficEnv, create_sumocfg  # Added create_sumocfg import
from core.sar_framework import (
    create_state_representation,
    create_action_strategy,
    create_reward_function
)

logger = logging.getLogger(__name__)

# Global SAR configuration (can be loaded from file)
DEFAULT_SAR_CONFIG = {
    'max_flow': 10000.0,
    'max_occupancy': 100.0,
    'max_queue_length': 575.0 * 3 / 7
}


class TrafficEnvCompat(TrafficEnv):
    """
    Backward-compatible TrafficEnv that accepts string parameters
    for state, action, and reward specifications.
    """
    
    def __init__(self,
                 port: int,
                 model_name: str,
                 model_idx: int,
                 sim_length: int,
                 base_gen_car_distrib: list,
                 num_of_episodes: int,
                 # Accept both string and object specifications
                 state_representation: Union[str, Any] = "full_metrics",
                 action_strategy: Union[str, Any] = "absolute_speed",
                 reward_fn: Union[str, Any] = "balanced",
                 vsl_enforcement: str = "recommend",
                 sumo_binary_path_override: Optional[str] = None,
                 normalization_bounds_path: Optional[str] = None,
                 sar_config: Optional[Dict[str, Any]] = None):
        
        # Load SAR configuration
        if sar_config is None:
            sar_config = DEFAULT_SAR_CONFIG.copy()
            
        # Load normalization bounds if provided
        if normalization_bounds_path and os.path.exists(normalization_bounds_path):
            try:
                import json
                with open(normalization_bounds_path, 'r') as f:
                    data = json.load(f)
                if "bounds" in data:
                    bounds = data["bounds"]
                    sar_config.update(bounds)
                    logger.info(f"Loaded normalization bounds from {normalization_bounds_path}")
            except Exception as e:
                logger.warning(f"Could not load bounds from {normalization_bounds_path}: {e}")
        
        # Convert string specifications to objects if needed
        if isinstance(state_representation, str):
            state_repr_obj = create_state_representation(state_representation, sar_config)
        else:
            state_repr_obj = state_representation
            
        if isinstance(action_strategy, str):
            action_strat_obj = create_action_strategy(action_strategy, sar_config)
        else:
            action_strat_obj = action_strategy
            
        if isinstance(reward_fn, str):
            reward_func_obj = create_reward_function(reward_fn, sar_config)
        else:
            reward_func_obj = reward_fn

        effective_model_name = f"{model_name}_{model_idx}"
        create_sumocfg(effective_model_name)
        
        # Initialize parent with objects
        super().__init__(
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
            sar_config=sar_config
        )


def create_train_env_compat(idx: int,
                           model_name: str,
                           sim_length: int,
                           num_of_episodes: int,
                           reward_fn: str,
                           vsl_enforcement: str = "recommend",
                           state_representation: str = "full_metrics",
                           action_strategy: str = "absolute_speed",
                           sumo_port: Optional[int] = None,
                           sumo_binary: Optional[str] = None,
                           normalization_bounds_path: Optional[str] = None):
    """
    Create a training environment with backward compatibility.
    This function can be used as a drop-in replacement for the original.
    """
    from stable_baselines3.common.monitor import Monitor
    
    port = sumo_port if sumo_port is not None else 8000 + idx
    
    # Auto-detect normalization bounds path if not provided
    if normalization_bounds_path is None:
        bounds_path = Path("rl_models/optuna_params") / f"best_optuna_hyperparams_{model_name}.json"
        if bounds_path.exists():
            normalization_bounds_path = str(bounds_path)
    
    env = TrafficEnvCompat(
        port=port,
        model_name=model_name,
        model_idx=idx,
        sim_length=sim_length,
        base_gen_car_distrib=["uniform", 2000],  # Default, will be overridden in reset
        num_of_episodes=num_of_episodes,
        state_representation=state_representation,
        action_strategy=action_strategy,
        reward_fn=reward_fn,
        vsl_enforcement=vsl_enforcement,
        sumo_binary_path_override=sumo_binary,
        normalization_bounds_path=normalization_bounds_path
    )
    
    return Monitor(env)


def create_eval_env_compat(model_name: str,
                          sim_length: int,
                          reward_fn: str,
                          vsl_enforcement: str = "recommend",
                          state_representation: str = "full_metrics",
                          action_strategy: str = "absolute_speed",
                          sumo_port: Optional[int] = None,
                          sumo_binary: Optional[str] = None,
                          normalization_bounds_path: Optional[str] = None,
                          eval_idx: int = 0):
    """
    Create an evaluation environment with backward compatibility.
    """
    from stable_baselines3.common.monitor import Monitor
    from gymnasium.wrappers import TimeLimit
    
    port = sumo_port if sumo_port is not None else 10000 + eval_idx
    
    # Auto-detect normalization bounds path if not provided
    if normalization_bounds_path is None:
        bounds_path = Path("rl_models/optuna_params") / f"best_optuna_hyperparams_{model_name}.json"
        if bounds_path.exists():
            normalization_bounds_path = str(bounds_path)
    
    env = TrafficEnvCompat(
        port=port,
        model_name=model_name,
        model_idx=eval_idx,
        sim_length=sim_length,
        base_gen_car_distrib=["uniform", 3000],  # Eval default
        num_of_episodes=1,
        state_representation=state_representation,
        action_strategy=action_strategy,
        reward_fn=reward_fn,
        vsl_enforcement=vsl_enforcement,
        sumo_binary_path_override=sumo_binary,
        normalization_bounds_path=normalization_bounds_path
    )
    
    # Wrap with time limit
    max_episode_steps = sim_length // 60  # Assuming 60s aggregation
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    
    return Monitor(env)


def update_train_model_minimal(original_train_model_func):
    """
    Decorator to update existing train_model function with minimal changes.
    
    Usage:
        @update_train_model_minimal
        def train_model(algorithm, reward_function, ...):
            # Your existing train_model code
    """
    
    def wrapper(algorithm: str,
                reward_function: str = "balanced",
                num_of_episodes: int = 200,
                hyperparams: Optional[dict] = None,
                vsl_enforcement: str = "recommend",
                state_representation: str = "full_metrics",
                action_strategy: str = "absolute_speed",
                process_train_base_port: Optional[int] = None,
                process_eval_base_port: Optional[int] = None,
                sumo_binary_to_use: Optional[str] = None):
        
        # Store original env constructor functions if they exist
        import sys
        module = sys.modules[original_train_model_func.__module__]
        
        # Temporarily replace environment constructors
        original_train_constructor = getattr(module, 'train_env_constructor', None)
        original_eval_constructor = getattr(module, 'eval_env_constructor', None)
        
        # Create new constructors that use SAR parameters
        def new_train_constructor(idx, model_name, sim_length, num_episodes, 
                                 reward_fn, vsl_mode, port=None, binary=None):
            return create_train_env_compat(
                idx, model_name, sim_length, num_episodes,
                reward_fn, vsl_mode, state_representation, action_strategy,
                port, binary
            )
        
        def new_eval_constructor(model_name, sim_length, reward_fn, vsl_mode,
                                port=None, binary=None):
            return create_eval_env_compat(
                model_name, sim_length, reward_fn, vsl_mode,
                state_representation, action_strategy, port, binary
            )
        
        # Replace constructors
        setattr(module, 'train_env_constructor', new_train_constructor)
        setattr(module, 'eval_env_constructor', new_eval_constructor)
        
        try:
            # Call original function
            result = original_train_model_func(
                algorithm=algorithm,
                reward_function=reward_function,
                num_of_episodes=num_of_episodes,
                hyperparams=hyperparams,
                vsl_enforcement=vsl_enforcement,
                process_train_base_port=process_train_base_port,
                process_eval_base_port=process_eval_base_port,
                sumo_binary_to_use=sumo_binary_to_use
            )
            return result
            
        finally:
            # Restore original constructors
            if original_train_constructor:
                setattr(module, 'train_env_constructor', original_train_constructor)
            if original_eval_constructor:
                setattr(module, 'eval_env_constructor', original_eval_constructor)
    
    return wrapper


# Convenience functions for common SAR combinations
def create_mobility_env(port: int, model_name: str, model_idx: int, 
                       sim_length: int, **kwargs):
    """Create environment optimized for mobility"""
    return TrafficEnvCompat(
        port=port,
        model_name=model_name,
        model_idx=model_idx,
        sim_length=sim_length,
        state_representation="full_metrics",
        action_strategy="absolute_speed",
        reward_fn="mobility",
        **kwargs
    )


def create_safety_env(port: int, model_name: str, model_idx: int,
                     sim_length: int, **kwargs):
    """Create environment optimized for safety"""
    return TrafficEnvCompat(
        port=port,
        model_name=model_name,
        model_idx=model_idx,
        sim_length=sim_length,
        state_representation="full_metrics",
        action_strategy="relative_speed",  # Gradual changes for safety
        reward_fn="safety",
        **kwargs
    )


def create_custom_env(port: int, model_name: str, model_idx: int,
                     sim_length: int, sar_spec: Dict[str, str], **kwargs):
    """
    Create environment with custom SAR specification.
    
    Args:
        sar_spec: Dict with keys 'state', 'action', 'reward'
        
    Example:
        env = create_custom_env(
            port=8000, model_name="test", model_idx=0, sim_length=3600,
            sar_spec={
                'state': 'minimal',
                'action': 'conservative',
                'reward': 'emission'
            },
            base_gen_car_distrib=["uniform", 3000],
            num_of_episodes=1
        )
    """
    return TrafficEnvCompat(
        port=port,
        model_name=model_name,
        model_idx=model_idx,
        sim_length=sim_length,
        state_representation=sar_spec.get('state', 'full_metrics'),
        action_strategy=sar_spec.get('action', 'absolute_speed'),
        reward_fn=sar_spec.get('reward', 'balanced'),
        **kwargs
    )


# Example usage in existing code with minimal changes
if __name__ == "__main__":
    # Example 1: Direct replacement
    from stable_baselines3.common.vec_env import SubprocVecEnv
    
    # Original code would have:
    # envs = SubprocVecEnv([
    #     lambda i=i: train_env_constructor(i, "DQN_test", 3600, 10, "balanced", "recommend")
    #     for i in range(4)
    # ])
    
    # New code with minimal change:
    envs = SubprocVecEnv([
        lambda i=i: create_train_env_compat(
            i, "DQN_test", 3600, 10, "balanced", "recommend",
            "full_metrics", "absolute_speed"  # Just add these parameters
        )
        for i in range(4)
    ])
    
    # Example 2: Using convenience functions
    mobility_env = create_mobility_env(
        port=8000,
        model_name="DQN_mobility_test",
        model_idx=0,
        sim_length=3600,
        base_gen_car_distrib=["uniform", 3000],
        num_of_episodes=1
    )
    
    print("Integration wrapper loaded successfully!")