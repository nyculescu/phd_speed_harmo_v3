# drl_vsl_train.py
"""
DRL-VSL Training Script with Modular SAR Framework and YAML Configuration

This script provides training functionality for the DRL-based Variable Speed Limit
control system using the modular State-Action-Reward framework.
Configuration is loaded from a YAML file for easy parameter management.
"""

import os
import sys
import logging
import argparse
import json
import glob
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import multiprocessing as mp
from datetime import datetime, timezone
import torch.nn as nn
from torch.cuda import is_available as cuda_available

# Stable Baselines 3
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.logger import configure

# Import the modular components
from core.sar_framework import (
    create_state_representation,
    create_action_strategy,
    create_reward_function
)
from core.drl_vsl import TrafficEnv
from core.drl_vsl import create_sumocfg
from traffic_environment.flow_gen import flow_generation_fix_num_veh, flow_generation, bimodal_distribution_24h

# Default configuration file path
DEFAULT_CONFIG_DIR = "config"
DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join("training", DEFAULT_CONFIG_DIR, "drl_vsl_train_config.yaml"))

# Global SAR configuration (can be loaded from file)
DEFAULT_SAR_CONFIG = {
    'max_flow': 10000.0,
    'max_occupancy': 100.0,
    'max_queue_length': 575.0 * 3 / 7
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration class that loads and validates YAML config."""
    
    def __init__(self, config_path: str):
        """Load configuration from YAML file."""
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self.data = yaml.safe_load(f)
        
        # Validate configuration
        self._validate()
        
        # Set logging level
        log_level = self.data.get('logging', {}).get('level', 'INFO')
        logging.getLogger().setLevel(getattr(logging, log_level))
    
    def _validate(self):
        """Validate configuration structure and values."""
        required_sections = ['sar_config', 'model', 'training', 'hyperparameters', 
                        'execution', 'environment', 'logging']
        
        for section in required_sections:
            if section not in self.data:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate SAR options
        valid_states = ['full_metrics', 'minimal', 'marvel']
        valid_actions = ['absolute_speed', 'relative_speed', 'marvel_speed']
        valid_rewards = ['mobility', 'safety', 'balanced', 'marvel']
        valid_enforcement = ['recommend', 'all_vehicles', 'cavs_only']
        valid_training_modes = ['single', 'selected', 'all_custom', 'all_rewards']
        
        # Validate default SAR config
        if self.data['sar_config']['state_representation'] not in valid_states:
            raise ValueError(f"Invalid state representation: {self.data['sar_config']['state_representation']}")
        if self.data['sar_config']['action_strategy'] not in valid_actions:
            raise ValueError(f"Invalid action strategy: {self.data['sar_config']['action_strategy']}")
        if self.data['sar_config']['reward_function'] not in valid_rewards:
            raise ValueError(f"Invalid reward function: {self.data['sar_config']['reward_function']}")
        if self.data['model']['vsl_enforcement'] not in valid_enforcement:
            raise ValueError(f"Invalid VSL enforcement: {self.data['model']['vsl_enforcement']}")
        
        # Validate training mode
        training_mode = self.data.get('execution', {}).get('training_mode', 'single')
        if training_mode not in valid_training_modes:
            raise ValueError(f"Invalid training_mode: {training_mode}. Valid options: {valid_training_modes}")
        
        # Validate custom combinations if present
        if 'custom_combinations' in self.data['sar_config']:
            for i, combo in enumerate(self.data['sar_config']['custom_combinations']):
                if 'name' not in combo or 'state' not in combo or 'action' not in combo or 'reward' not in combo:
                    raise ValueError(f"Invalid custom combination at index {i}: missing required fields")
                if combo['state'] not in valid_states:
                    raise ValueError(f"Invalid state '{combo['state']}' in custom combination '{combo['name']}'")
                if combo['action'] not in valid_actions:
                    raise ValueError(f"Invalid action '{combo['action']}' in custom combination '{combo['name']}'")
                if combo['reward'] not in valid_rewards:
                    raise ValueError(f"Invalid reward '{combo['reward']}' in custom combination '{combo['name']}'")
        
        # Validate selected_combinations if training_mode is 'selected'
        if training_mode == 'selected':
            selected = self.data.get('execution', {}).get('selected_combinations')
            if selected is None:
                raise ValueError("training_mode is 'selected' but selected_combinations is not specified")
            
            # Validate that selections are valid (string names or integer indices)
            custom_combos = self.data.get('sar_config', {}).get('custom_combinations', [])
            selections = selected if isinstance(selected, list) else [selected]
            
            for sel in selections:
                if isinstance(sel, str):
                    # Check if name exists
                    if not any(c['name'] == sel for c in custom_combos):
                        available = [c['name'] for c in custom_combos]
                        raise ValueError(f"Selected combination '{sel}' not found. Available: {available}")
                elif isinstance(sel, int):
                    # Check if index is valid
                    if not (0 <= sel < len(custom_combos)):
                        raise ValueError(f"Selected index {sel} out of range. Valid range: 0-{len(custom_combos)-1}")
                else:
                    raise ValueError(f"Invalid selection type: {type(sel)}. Use string name or integer index")
    
    def get(self, key_path: str, default=None):
        """Get nested configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value


def get_linear_schedule(initial_value: float):
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def load_hyperparameters(config: Config, model_name: str) -> Dict[str, Any]:
    """Load hyperparameters from config and Optuna if available."""
    
    # Start with defaults from config
    hyperparams = config.get('hyperparameters.defaults', {}).copy()
    
    # Handle activation function
    if 'policy_kwargs' in hyperparams:
        if hyperparams['policy_kwargs'].get('activation_fn') == 'ReLU':
            hyperparams['policy_kwargs']['activation_fn'] = nn.ReLU
    
    # Convert train_freq to tuple if needed
    if 'train_freq' in hyperparams:
        if isinstance(hyperparams['train_freq'], list):
            hyperparams['train_freq'] = tuple(hyperparams['train_freq'])
    
    # Try to load Optuna hyperparameters if enabled
    if config.get('hyperparameters.use_optuna', False):
        optuna_dir = Path("rl_models/optuna_params")
        optuna_path = optuna_dir / f"best_optuna_hyperparams_{model_name}.json"
        
        if optuna_path.exists():
            try:
                with open(optuna_path, 'r') as f:
                    data = json.load(f)
                
                algorithm = config.get('model.algorithm', 'DQN')
                if algorithm in data:
                    optuna_params = data[algorithm]
                    
                    # Process policy kwargs
                    if "policy_kwargs" in optuna_params:
                        if "activation_fn" in optuna_params["policy_kwargs"]:
                            if optuna_params["policy_kwargs"]["activation_fn"] == "nn.ReLU":
                                optuna_params["policy_kwargs"]["activation_fn"] = nn.ReLU
                    
                    # Process train_freq
                    if "train_freq" in optuna_params:
                        if isinstance(optuna_params["train_freq"], list):
                            optuna_params["train_freq"] = tuple(optuna_params["train_freq"])
                        elif isinstance(optuna_params["train_freq"], int):
                            optuna_params["train_freq"] = (optuna_params["train_freq"], "step")
                    
                    # Update hyperparams with Optuna values
                    hyperparams.update(optuna_params)
                    logger.info(f"Loaded Optuna hyperparameters from {optuna_path}")
                    
            except Exception as e:
                logger.warning(f"Could not load Optuna params: {e}. Using defaults.")
    
    # Apply manual overrides from config
    overrides = config.get('hyperparameters.overrides', {})
    for key, value in overrides.items():
        if value is not None:
            hyperparams[key] = value
            logger.info(f"Applied manual override: {key} = {value}")
    
    # Apply learning rate schedule if needed
    if isinstance(hyperparams.get("learning_rate"), (int, float)):
        hyperparams["learning_rate"] = get_linear_schedule(hyperparams["learning_rate"])
    
    return hyperparams


def load_sar_config(config: Config, model_name: str) -> Dict[str, Any]:
    """Load SAR configuration including normalization bounds."""
    
    sar_config = {
        'max_flow': config.get('environment.normalization.max_flow', 10000.0),
        'max_occupancy': config.get('environment.normalization.max_occupancy', 100.0),
        'max_queue_length': config.get('environment.normalization.max_queue_length', 246.4)
    }
    
    # Try to load bounds from Optuna file if available
    if config.get('hyperparameters.use_optuna', False):
        optuna_dir = Path("rl_models/optuna_params")
        bounds_path = optuna_dir / f"best_optuna_hyperparams_{model_name}.json"
        
        if bounds_path.exists():
            try:
                with open(bounds_path, 'r') as f:
                    data = json.load(f)
                    
                if "bounds" in data:
                    sar_config.update(data["bounds"])
                    logger.info(f"Loaded normalization bounds from {bounds_path}")
                    
            except Exception as e:
                logger.warning(f"Could not load bounds: {e}. Using defaults.")
    
    return sar_config


def create_train_env(env_idx: int,
                     config: Config,
                     model_name: str,
                     sar_config: Dict[str, Any],
                     port: int,
                     state_repr: str,
                     action_strat: str,
                     reward_func: str) -> Monitor:
    """Create a training environment with SAR components."""
    
    # Create SAR components
    state_repr_obj = create_state_representation(state_repr, sar_config)
    action_strat_obj = create_action_strategy(action_strat, sar_config)
    reward_func_obj = create_reward_function(reward_func, sar_config)
    
    # Get SUMO configuration
    sumo_config = None
    sumo_preset = None
    
    sumo_mode = config.get('sumo.config_mode', 'default')
    
    if sumo_mode == 'preset':
        sumo_preset = config.get('sumo.preset_name')
        logger.info(f"Using SUMO preset: {sumo_preset}")
    elif sumo_mode == 'file':
        sumo_config = config.get('sumo.config_file')
        logger.info(f"Using SUMO config file: {sumo_config}")
    elif sumo_mode == 'custom':
        sumo_config = config.get('sumo.custom_config', {})
        logger.info("Using custom SUMO configuration from YAML")
    # else: default mode, both remain None
    
    # Create environment
    env = TrafficEnv(
        port=port,
        model_name=model_name,
        model_idx=env_idx,
        sim_length=config.get('training.train_sim_length', 7200),
        base_gen_car_distrib=["uniform", 2000],  # Will be overridden in reset
        num_of_episodes=config.get('training.num_episodes', 200),
        state_representation=state_repr_obj,
        action_strategy=action_strat_obj,
        reward_function=reward_func_obj,
        vsl_enforcement=config.get('model.vsl_enforcement'),
        sumo_binary_path_override=config.get('environment.sumo_binary'),
        sar_config=sar_config,
        sumo_config=sumo_config,
        sumo_preset=sumo_preset
    )
    
    return Monitor(env)


def create_eval_env(config: Config,
                    model_name: str,
                    sar_config: Dict[str, Any],
                    port: int,
                    state_repr: str,
                    action_strat: str,
                    reward_func: str) -> Monitor:
    """Create an evaluation environment with SAR components."""
    
    from gymnasium.wrappers import TimeLimit
    
    # Create SAR components
    state_repr_obj = create_state_representation(state_repr, sar_config)
    action_strat_obj = create_action_strategy(action_strat, sar_config)
    reward_func_obj = create_reward_function(reward_func, sar_config)
    
    # Get SUMO configuration
    sumo_config = None
    sumo_preset = None
    
    sumo_mode = config.get('sumo.config_mode', 'default')
    
    if sumo_mode == 'preset':
        sumo_preset = config.get('sumo.preset_name')
        logger.info(f"Using SUMO preset for eval: {sumo_preset}")
    elif sumo_mode == 'file':
        sumo_config = config.get('sumo.config_file')
        logger.info(f"Using SUMO config file for eval: {sumo_config}")
    elif sumo_mode == 'custom':
        sumo_config = config.get('sumo.custom_config', {})
        logger.info("Using custom SUMO configuration from YAML for eval")
    # else: default mode, both remain None
    
    eval_sim_length = config.get('training.eval_sim_length', 14400)
    
    # Create environment
    env = TrafficEnv(
        port=port,
        model_name=model_name,
        model_idx=0,  # Eval env uses idx 0
        sim_length=eval_sim_length,
        base_gen_car_distrib=["uniform", 3000],  # Fixed eval scenario
        num_of_episodes=1,
        state_representation=state_repr_obj,
        action_strategy=action_strat_obj,
        reward_function=reward_func_obj,
        vsl_enforcement=config.get('model.vsl_enforcement'),
        sumo_binary_path_override=config.get('environment.sumo_binary'),
        sar_config=sar_config,
        sumo_config=sumo_config,
        sumo_preset=sumo_preset
    )
    
    # Wrap with time limit
    max_episode_steps = eval_sim_length // 60  # Assuming 60s aggregation
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    
    return Monitor(env)


def train_model(config: Config,
                state_representation: Optional[str] = None,
                action_strategy: Optional[str] = None,
                reward_function: Optional[str] = None,
                process_base_port: Optional[int] = None,
                custom_model_name: Optional[str] = None):
    """
    Train a DRL model with specified configuration.
    
    Args:
        config: Configuration object
        state_representation: Override state representation from config
        action_strategy: Override action strategy from config
        reward_function: Override reward function from config
        process_base_port: Base port for SUMO instances
        custom_model_name: Custom name for the model (overrides auto-generated name)
    """
    
    # Get SAR configuration (use overrides if provided)
    state_repr = state_representation or config.get('sar_config.state_representation')
    action_strat = action_strategy or config.get('sar_config.action_strategy')
    reward_func = reward_function or config.get('sar_config.reward_function')
    
    # Get model configuration
    algorithm = config.get('model.algorithm', 'DQN')
    vsl_enforcement = config.get('model.vsl_enforcement')
    
    # Model naming - use custom name if provided
    if custom_model_name:
        full_model_name = f"{algorithm}_{custom_model_name}_{vsl_enforcement}"
        model_name = f"{algorithm}_{custom_model_name}"
    else:
        model_name = f"{algorithm}_{reward_func}_{vsl_enforcement}"
        full_model_name = f"{model_name}_{state_repr}_{action_strat}"
        
        # Override with config model name if provided
        if config.get('model.model_name'):
            full_model_name = config.get('model.model_name')
    
    # Setup directories
    log_base = Path(config.get('logging.log_dir', './logs'))
    model_base = Path(config.get('logging.model_dir', './rl_models'))
    
    log_dir = log_base / full_model_name
    model_dir = model_base / full_model_name
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting training for {full_model_name}")
    logger.info(f"SAR Configuration: State={state_repr}, Action={action_strat}, Reward={reward_func}")
    
    # Load configurations
    sar_config = load_sar_config(config, model_name)
    hyperparams = load_hyperparameters(config, model_name)
    
    # Set up ports
    train_base_port = process_base_port or config.get('environment.train_base_port', 8000)
    num_train_envs = config.get('training.num_train_envs', 4)
    eval_base_port = train_base_port + num_train_envs + 10
    
    # Create SUMO config files for all environments
    for i in range(num_train_envs):
        create_sumocfg(f"{model_name}_{i}")
    
    # Create eval config
    create_sumocfg(f"{model_name}_eval_0")
    
    # Create training environments
    logger.info(f"Creating {num_train_envs} training environments...")
    train_env = SubprocVecEnv([
        lambda i=i: create_train_env(
            i, config, model_name, sar_config, train_base_port + i,
            state_repr, action_strat, reward_func
        )
        for i in range(num_train_envs)
    ])
    
    # Create evaluation environment
    logger.info("Creating evaluation environment...")
    eval_env = SubprocVecEnv([
        lambda: create_eval_env(
            config, f"{model_name}_eval", sar_config, eval_base_port,
            state_repr, action_strat, reward_func
        )
    ])
    
    # Extract policy kwargs and other params
    policy_kwargs = hyperparams.pop("policy_kwargs", {"net_arch": [256, 256, 128], "activation_fn": nn.ReLU})
    
    # Determine device
    device_config = config.get('execution.device', 'auto')
    if device_config == 'auto':
        device = 'cuda' if cuda_available() else 'cpu'
    else:
        device = device_config
    
    # Create a copy for logging with readable values
    log_hyperparams = hyperparams.copy()
    if callable(hyperparams.get("learning_rate")):
        # Extract the initial learning rate value from the schedule function
        # The schedule function is created with the initial value in its closure
        initial_lr = config.get('hyperparameters.overrides.learning_rate') or \
                    config.get('hyperparameters.defaults.learning_rate', 0.0001)
        log_hyperparams["learning_rate"] = f"{initial_lr} (linear schedule)"
        
    logger.info(f"Creating {algorithm} model with hyperparameters:")
    logger.info(json.dumps({k: str(v) for k, v in log_hyperparams.items()}, indent=2))
    
    model = DQN(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=config.get('logging.verbose', 1),
        tensorboard_log=str(log_dir) if config.get('logging.tensorboard', True) else None,
        device=device,
        **hyperparams
    )
    
    # Set random seed if specified
    if config.get('advanced.seed') is not None:
        model.set_random_seed(config.get('advanced.seed'))
    
    # Set up logger
    log_formats = ["stdout"]
    if config.get('logging.csv_logging', True):
        log_formats.append("csv")
    if config.get('logging.tensorboard', True):
        log_formats.append("tensorboard")
    
    model.set_logger(configure(str(log_dir), log_formats))
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=config.get('training.checkpoint_freq', 25000),
        save_path=str(model_dir),
        name_prefix=f"rl_model_{full_model_name}",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1
    )
    
    callbacks = [checkpoint_callback]
    
    # Early stopping callback
    if config.get('advanced.early_stopping.enabled', True):
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=config.get('advanced.early_stopping.max_no_improvement_evals', 10),
            min_evals=config.get('advanced.early_stopping.min_evals', 5),
            verbose=1
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(log_dir),
            eval_freq=config.get('training.eval_freq', 25000),
            n_eval_episodes=1,
            deterministic=True,
            render=False,
            callback_after_eval=stop_callback,
            verbose=1
        )
        callbacks.append(eval_callback)
    else:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir),
            log_path=str(log_dir),
            eval_freq=config.get('training.eval_freq', 25000),
            n_eval_episodes=1,
            deterministic=True,
            render=False,
            verbose=1
        )
        callbacks.append(eval_callback)
    
    # Training
    try:
        total_timesteps = config.get('training.total_timesteps', 500000)
        logger.info(f"Starting training for {total_timesteps} timesteps...")
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=config.get('execution.progress_bar', True),
            reset_num_timesteps=False
        )
        
        # Save final model
        final_path = model_dir / f"{full_model_name}_final.zip"
        model.save(str(final_path))
        logger.info(f"Training completed! Final model saved to {final_path}")
        
    except KeyboardInterrupt:
        logger.warning(f"Training interrupted by user for {full_model_name}")
        model.save(str(model_dir / f"{full_model_name}_interrupted.zip"))
    
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise
    
    finally:
        train_env.close()
        eval_env.close()
        logger.info(f"Finished training for {full_model_name}")


def run_parallel_training(config: Config, configurations: list, custom_names: Optional[list] = None):
    """
    Run multiple training configurations in parallel.
    
    Args:
        config: Configuration object
        configurations: List of (state, action, reward) tuples
        custom_names: Optional list of custom names for each configuration
    """
    
    num_processes = config.get('execution.num_processes')
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)
    
    logger.info(f"Running {len(configurations)} training configurations using {num_processes} processes")
    
    # Prepare arguments for each configuration
    training_args = []
    base_port = config.get('environment.train_base_port', 8000)
    
    for i, (state, action, reward) in enumerate(configurations):
        args = {
            'config': config,
            'state_representation': state,
            'action_strategy': action,
            'reward_function': reward,
            'process_base_port': base_port + i * 100,
            'custom_model_name': custom_names[i] if custom_names else None
        }
        training_args.append(args)
    
    # Run training in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(train_model_wrapper, training_args)
    
    # Report results
    logger.info("\nTraining Summary:")
    for i, (conf, result) in enumerate(zip(configurations, results)):
        name = custom_names[i] if custom_names else str(conf)
        status = "Success" if result else "Failed"
        logger.info(f"  {name}: {status}")


def train_model_wrapper(args_dict):
    """Wrapper for multiprocessing pool."""
    try:
        train_model(**args_dict)
        return True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False


def cleanup_temp_files(model_name: str):
    """Clean up temporary SUMO files after training."""
    patterns = [
        f"./traffic_environment/sumo/generated_flows/*{model_name}*.rou.xml",
        f"./traffic_environment/sumo/generated_configs/*{model_name}*.sumocfg",
        f"./logs/sumo_log/*{model_name}*.txt"
    ]
    
    for pattern in patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                logger.debug(f"Removed temp file: {file}")
            except Exception as e:
                logger.warning(f"Could not remove {file}: {e}")


def determine_configurations(config: Config) -> Tuple[list, list]:
    """
    Determine which configurations to train based on config settings.
    
    Returns:
        Tuple of (configurations, custom_names) where:
        - configurations: List of (state, action, reward) tuples
        - custom_names: List of custom names or None
    """
    
    training_mode = config.get('execution.training_mode', 'single')
    custom_combos = config.get('sar_config.custom_combinations', [])
    
    if training_mode == 'selected':
        # Train selected combinations only
        selected = config.get('execution.selected_combinations')
        
        if selected is None:
            logger.error("training_mode is 'selected' but no selected_combinations specified")
            sys.exit(1)
        
        # Normalize to list
        if not isinstance(selected, list):
            selected = [selected]
        
        configurations = []
        custom_names = []
        
        for selection in selected:
            if isinstance(selection, str):
                # Selection by name
                found = False
                for combo in custom_combos:
                    if combo['name'] == selection:
                        configurations.append((combo['state'], combo['action'], combo['reward']))
                        custom_names.append(combo['name'])
                        found = True
                        break
                
                if not found:
                    available = [c['name'] for c in custom_combos]
                    logger.error(f"\nConfiguration Error:")
                    logger.error(f"  Combination '{selection}' not found in custom_combinations")
                    logger.error(f"\nAvailable combinations:")
                    for i, name in enumerate(available):
                        logger.error(f"    [{i}] {name}")
                    logger.error(f"\nPlease update selected_combinations in your YAML config")
                    sys.exit(1)
                    
            elif isinstance(selection, int):
                # Selection by index
                if 0 <= selection < len(custom_combos):
                    combo = custom_combos[selection]
                    configurations.append((combo['state'], combo['action'], combo['reward']))
                    custom_names.append(combo['name'])
                else:
                    logger.error(f"Index {selection} out of range. Valid range: 0-{len(custom_combos)-1}")
                    sys.exit(1)
            else:
                logger.error(f"Invalid selection type: {type(selection)}. Use string name or integer index.")
                sys.exit(1)
        
        logger.info(f"Selected {len(configurations)} combination(s) to train")
        return configurations, custom_names
    
    elif training_mode == 'all_custom':
        # Train all custom combinations
        if not custom_combos:
            logger.error("training_mode is 'all_custom' but no custom_combinations defined")
            sys.exit(1)
        
        configurations = []
        custom_names = []
        for combo in custom_combos:
            configurations.append((combo['state'], combo['action'], combo['reward']))
            custom_names.append(combo['name'])
        return configurations, custom_names
    
    elif training_mode == 'all_rewards':
        # All reward functions with default state/action
        base_state = config.get('sar_config.state_representation')
        base_action = config.get('sar_config.action_strategy')
        configurations = [
            (base_state, base_action, 'mobility'),
            (base_state, base_action, 'safety'),
            (base_state, base_action, 'balanced'),
            (base_state, base_action, 'marvel'),
        ]
        return configurations, None
    
    elif training_mode == 'single':
        # Single configuration from default settings
        configurations = [(
            config.get('sar_config.state_representation'),
            config.get('sar_config.action_strategy'),
            config.get('sar_config.reward_function')
        )]
        return configurations, None
    
    else:
        logger.error(f"Unknown training mode: {training_mode}")
        logger.info("Valid modes: 'single', 'selected', 'all_custom', 'all_rewards'")
        sys.exit(1)


def main():
    """Main entry point for training script."""
    
    parser = argparse.ArgumentParser(
        description="Train DRL-VSL models with YAML configuration",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH,
                       help=f'Path to YAML configuration file (default: {DEFAULT_CONFIG_PATH})')
    
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate the configuration file without training')
    
    parser.add_argument('--list-combos', action='store_true',
                       help='List available custom combinations and exit')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = Config(args.config)
        logger.info(f"Successfully loaded configuration from {args.config}")
        
        if args.validate_only:
            logger.info("Configuration validation successful!")
            return
            
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # List combinations if requested
    if args.list_combos:
        custom_combos = config.get('sar_config.custom_combinations', [])
        if not custom_combos:
            logger.info("No custom combinations defined in configuration.")
        else:
            logger.info("\nAvailable custom combinations:")
            for i, combo in enumerate(custom_combos):
                logger.info(f"  [{i}] {combo['name']:<25} - state: {combo['state']:<15} "
                           f"action: {combo['action']:<15} reward: {combo['reward']}")
        return
    
    # Check SUMO
    if 'SUMO_HOME' not in os.environ:
        logger.error("Please set SUMO_HOME environment variable")
        sys.exit(1)
    
    # Validate training mode and selections
    training_mode = config.get('execution.training_mode', 'single')
    if training_mode == 'selected':
        selected = config.get('execution.selected_combinations')
        if selected is None:
            logger.error("\nConfiguration Error:")
            logger.error("  training_mode is 'selected' but selected_combinations is not specified")
            logger.error("\nPlease add to your YAML config:")
            logger.error("  execution:")
            logger.error("    selected_combinations: \"experiment_name\"  # or [\"name1\", \"name2\"]")
            sys.exit(1)
    
    # Determine configurations based on YAML settings
    configurations, custom_names = determine_configurations(config)
    
    # Log training plan
    logger.info("\nTraining Plan:")
    logger.info(f"  Algorithm: {config.get('model.algorithm')}")
    logger.info(f"  VSL Enforcement: {config.get('model.vsl_enforcement')}")
    logger.info(f"  Total Timesteps: {config.get('training.total_timesteps'):,}")
    logger.info(f"  Training Mode: {config.get('execution.training_mode')}")
    logger.info(f"  Configurations to train: {len(configurations)}")
    
    for i, (state, action, reward) in enumerate(configurations, 1):
        name = custom_names[i-1] if custom_names else f"{state}_{action}_{reward}"
        logger.info(f"    {i}. {name:<25} - State: {state:<15} Action: {action:<15} Reward: {reward}")
    
    # Run training
    if config.get('execution.parallel', False) and len(configurations) > 1:
        run_parallel_training(config, configurations, custom_names)
    else:
        for i, (state, action, reward) in enumerate(configurations):
            custom_name = custom_names[i] if custom_names else None
            train_model(
                config=config,
                state_representation=state,
                action_strategy=action,
                reward_function=reward,
                custom_model_name=custom_name
            )
            
            if config.get('execution.cleanup', True):
                algorithm = config.get('model.algorithm')
                vsl_enforcement = config.get('model.vsl_enforcement')
                if custom_name:
                    model_name = f"{algorithm}_{custom_name}"
                else:
                    model_name = f"{algorithm}_{reward}_{vsl_enforcement}"
                cleanup_temp_files(model_name)
    
    logger.info("\nAll training completed successfully!")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()