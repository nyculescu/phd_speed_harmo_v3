# drl_vsl_train.py
"""
DRL-VSL Training Script with Modular SAR Framework

This script provides training functionality for the DRL-based Variable Speed Limit
control system using the modular State-Action-Reward framework.
"""

import os
import sys
import logging
import argparse
import json
import glob
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
from core.drl_vsl_refactored import TrafficEnv
from core.drl_vsl_integration import TrafficEnvCompat
from core.drl_vsl_refactored import create_sumocfg
from traffic_environment.flow_gen import flow_generation_fix_num_veh, flow_generation, bimodal_distribution_24h

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_TRAIN_SUMO_PORT = 8000
BASE_EVAL_SUMO_PORT = 10000
PROGRESS_BAR_ENABLED = True
OPTUNA_PARAMS_DIR = Path("rl_models/optuna_params")
SUMO_CONFIG_DIR = Path("./traffic_environment/sumo")
CAVS_PRESENCE_PERCENTAGE = 10

# Default hyperparameters
DEFAULT_HYPERPARAMS = {
    "DQN": {
        "policy_kwargs": {
            "net_arch": [256, 256, 128],
            "activation_fn": nn.ReLU,
        },
        "learning_rate": 1e-4,
        "buffer_size": int(2e5),
        "batch_size": 32,
        "gamma": 0.995,
        "tau": 0.01,
        "exploration_fraction": 0.2,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.02,
        "learning_starts": 20000,
        "train_freq": (2, "step"),
        "target_update_interval": 2000,
        "gradient_steps": 2,
    }
}

# Default SAR configuration
DEFAULT_SAR_CONFIG = {
    'max_flow': 10000.0,
    'max_occupancy': 100.0,
    'max_queue_length': 575.0 * 3 / 7
}


def get_linear_schedule(initial_value: float):
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def load_hyperparameters(model_name: str, algorithm: str = "DQN") -> Dict[str, Any]:
    """Load hyperparameters from Optuna results or use defaults."""
    
    # Start with defaults
    hyperparams = DEFAULT_HYPERPARAMS.get(algorithm, DEFAULT_HYPERPARAMS["DQN"]).copy()
    
    # Try to load Optuna hyperparameters
    optuna_path = OPTUNA_PARAMS_DIR / f"best_optuna_hyperparams_{model_name}.json"
    if optuna_path.exists():
        try:
            with open(optuna_path, 'r') as f:
                data = json.load(f)
                
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
    
    # Apply learning rate schedule if needed
    if isinstance(hyperparams.get("learning_rate"), (int, float)):
        hyperparams["learning_rate"] = get_linear_schedule(hyperparams["learning_rate"])
    
    return hyperparams


def load_sar_config(model_name: str) -> Dict[str, Any]:
    """Load SAR configuration including normalization bounds."""
    
    config = DEFAULT_SAR_CONFIG.copy()
    
    # Try to load bounds from Optuna file
    bounds_path = OPTUNA_PARAMS_DIR / f"best_optuna_hyperparams_{model_name}.json"
    if bounds_path.exists():
        try:
            with open(bounds_path, 'r') as f:
                data = json.load(f)
                
            if "bounds" in data:
                config.update(data["bounds"])
                logger.info(f"Loaded normalization bounds from {bounds_path}")
                
        except Exception as e:
            logger.warning(f"Could not load bounds: {e}. Using defaults.")
    
    return config


def create_train_env(env_idx: int,
                     model_name: str,
                     sim_length: int,
                     num_episodes: int,
                     state_name: str,
                     action_name: str,
                     reward_name: str,
                     vsl_enforcement: str,
                     port: int,
                     sumo_binary: Optional[str] = None,
                     sar_config: Optional[Dict[str, Any]] = None) -> Monitor:
    """Create a training environment with SAR components."""
    
    # Create SAR components
    state_repr = create_state_representation(state_name, sar_config or DEFAULT_SAR_CONFIG)
    action_strat = create_action_strategy(action_name, sar_config or DEFAULT_SAR_CONFIG)
    reward_func = create_reward_function(reward_name, sar_config or DEFAULT_SAR_CONFIG)
    
    # Create environment
    env = TrafficEnv(
        port=port,
        model_name=model_name,
        model_idx=env_idx,
        sim_length=sim_length,
        base_gen_car_distrib=["uniform", 2000],  # Will be overridden in reset
        num_of_episodes=num_episodes,
        state_representation=state_repr,
        action_strategy=action_strat,
        reward_function=reward_func,
        vsl_enforcement=vsl_enforcement,
        sumo_binary_path_override=sumo_binary,
        sar_config=sar_config
    )
    
    return Monitor(env)


def create_eval_env(model_name: str,
                    sim_length: int,
                    state_name: str,
                    action_name: str,
                    reward_name: str,
                    vsl_enforcement: str,
                    port: int,
                    sumo_binary: Optional[str] = None,
                    sar_config: Optional[Dict[str, Any]] = None) -> Monitor:
    """Create an evaluation environment with SAR components."""
    
    from gymnasium.wrappers import TimeLimit
    
    # Create SAR components
    state_repr = create_state_representation(state_name, sar_config or DEFAULT_SAR_CONFIG)
    action_strat = create_action_strategy(action_name, sar_config or DEFAULT_SAR_CONFIG)
    reward_func = create_reward_function(reward_name, sar_config or DEFAULT_SAR_CONFIG)
    
    # Create environment
    env = TrafficEnv(
        port=port,
        model_name=model_name,
        model_idx=0,  # Eval env uses idx 0
        sim_length=sim_length,
        base_gen_car_distrib=["uniform", 3000],  # Fixed eval scenario
        num_of_episodes=1,
        state_representation=state_repr,
        action_strategy=action_strat,
        reward_function=reward_func,
        vsl_enforcement=vsl_enforcement,
        sumo_binary_path_override=sumo_binary,
        sar_config=sar_config
    )
    
    # Wrap with time limit
    max_episode_steps = sim_length // 60  # Assuming 60s aggregation
    env = TimeLimit(env, max_episode_steps=max_episode_steps)
    
    return Monitor(env)


def train_model(algorithm: str = "DQN",
                state_representation: str = "full_metrics",
                action_strategy: str = "absolute_speed",
                reward_function: str = "balanced",
                vsl_enforcement: str = "recommend",
                total_timesteps: int = 500_000,
                num_train_envs: int = 4,
                num_episodes: int = 200,
                eval_freq: int = 25_000,
                checkpoint_freq: int = 25_000,
                use_optuna_hyperparams: bool = True,
                process_base_port: Optional[int] = None,
                sumo_binary: Optional[str] = None,
                custom_hyperparams: Optional[Dict[str, Any]] = None):
    """
    Train a DRL model with specified SAR configuration.
    
    Args:
        algorithm: RL algorithm to use (currently only "DQN")
        state_representation: Name of state representation to use
        action_strategy: Name of action strategy to use
        reward_function: Name of reward function to use
        vsl_enforcement: VSL enforcement mode ("recommend", "all_vehicles", "cavs_only")
        total_timesteps: Total training timesteps
        num_train_envs: Number of parallel training environments
        num_episodes: Number of episodes per environment
        eval_freq: Evaluation frequency (steps per env)
        checkpoint_freq: Checkpoint save frequency
        use_optuna_hyperparams: Whether to load Optuna-tuned hyperparameters
        process_base_port: Base port for SUMO instances
        sumo_binary: Path to SUMO binary
        custom_hyperparams: Optional custom hyperparameters
    """
    
    # Model naming
    model_name = f"{algorithm}_{reward_function}_{vsl_enforcement}"
    full_model_name = f"{model_name}_{state_representation}_{action_strategy}"
    
    # Setup directories
    log_dir = Path(f"./logs/{full_model_name}/")
    model_dir = Path(f"./rl_models/{full_model_name}/")
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting training for {full_model_name}")
    logger.info(f"SAR Configuration: State={state_representation}, Action={action_strategy}, Reward={reward_function}")
    
    # Load configurations
    sar_config = load_sar_config(model_name)
    
    if custom_hyperparams:
        hyperparams = custom_hyperparams
    elif use_optuna_hyperparams:
        hyperparams = load_hyperparameters(model_name, algorithm)
    else:
        hyperparams = DEFAULT_HYPERPARAMS[algorithm].copy()
    
    # Set up ports
    train_base_port = process_base_port if process_base_port else BASE_TRAIN_SUMO_PORT
    eval_base_port = train_base_port + num_train_envs + 10
    
    # Simulation lengths
    train_sim_length = 3600 * 2  # 2 hours for training
    eval_sim_length = 3600 * 4   # 4 hours for evaluation
    
    # Create SUMO config files for all environments
    for i in range(num_train_envs):
        create_sumocfg(f"{model_name}_{i}")
    
    # Create eval config
    create_sumocfg(f"{model_name}_eval")
    
    # Create training environments
    logger.info(f"Creating {num_train_envs} training environments...")
    train_env = SubprocVecEnv([
        lambda i=i: create_train_env(
            i, model_name, train_sim_length, num_episodes,
            state_representation, action_strategy, reward_function,
            vsl_enforcement, train_base_port + i, sumo_binary, sar_config
        )
        for i in range(num_train_envs)
    ])
    
    # Create evaluation environment
    logger.info("Creating evaluation environment...")
    eval_env = SubprocVecEnv([
        lambda: create_eval_env(
            f"{model_name}_eval", eval_sim_length,
            state_representation, action_strategy, reward_function,
            vsl_enforcement, eval_base_port, sumo_binary, sar_config
        )
    ])
    
    # Extract policy kwargs and other params
    policy_kwargs = hyperparams.pop("policy_kwargs", {"net_arch": [256, 256, 128], "activation_fn": nn.ReLU})
    
    # Create model
    logger.info(f"Creating {algorithm} model with hyperparameters:")
    logger.info(json.dumps({k: str(v) for k, v in hyperparams.items()}, indent=2))
    
    model = DQN(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=str(log_dir),
        device='cuda' if cuda_available() else 'cpu',
        **hyperparams
    )
    
    # Set up logger
    model.set_logger(configure(str(log_dir), ["stdout", "csv", "tensorboard"]))
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(model_dir),
        name_prefix=f"rl_model_{full_model_name}",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1
    )
    
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=5,
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(log_dir),
        eval_freq=eval_freq,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        callback_after_eval=stop_callback,
        verbose=1
    )
    
    # Training
    try:
        logger.info(f"Starting training for {total_timesteps} timesteps...")
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=PROGRESS_BAR_ENABLED,
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


def run_parallel_training(configurations: list,
                         num_processes: Optional[int] = None,
                         **kwargs):
    """
    Run multiple training configurations in parallel.
    
    Args:
        configurations: List of (state, action, reward) tuples
        num_processes: Number of parallel processes (default: CPU count - 1)
        **kwargs: Additional arguments passed to train_model
    """
    
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)
    
    logger.info(f"Running {len(configurations)} training configurations using {num_processes} processes")
    
    # Prepare arguments for each configuration
    training_args = []
    for i, (state, action, reward) in enumerate(configurations):
        args = {
            'state_representation': state,
            'action_strategy': action,
            'reward_function': reward,
            'process_base_port': BASE_TRAIN_SUMO_PORT + i * 100,
            **kwargs
        }
        training_args.append(args)
    
    # Run training in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(train_model_wrapper, training_args)
    
    # Report results
    logger.info("\nTraining Summary:")
    for config, result in zip(configurations, results):
        status = "Success" if result else "Failed"
        logger.info(f"  {config}: {status}")


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
        f"./traffic_environment/sumo/*{model_name}*.rou.xml",
        f"./traffic_environment/sumo/*{model_name}*.sumocfg",
        f"./logs/sumo_log/*{model_name}*.txt"
    ]
    
    for pattern in patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                logger.debug(f"Removed temp file: {file}")
            except Exception as e:
                logger.warning(f"Could not remove {file}: {e}")


def main():
    """Main entry point for training script."""
    
    parser = argparse.ArgumentParser(
        description="Train DRL-VSL models with modular SAR framework",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # SAR configuration
    parser.add_argument('--state', type=str, default='full_metrics',
                       choices=['full_metrics', 'minimal'],
                       help='State representation to use')
    parser.add_argument('--action', type=str, default='absolute_speed',
                       choices=['absolute_speed', 'relative_speed'],
                       help='Action strategy to use')
    parser.add_argument('--reward', type=str, default='balanced',
                       choices=['mobility', 'safety', 'balanced'],
                       help='Reward function to use')
    
    # Model configuration
    parser.add_argument('--algo', type=str, default='DQN',
                       choices=['DQN'],
                       help='RL algorithm to use')
    parser.add_argument('--vsl-mode', type=str, default='recommend',
                       choices=['recommend', 'all_vehicles', 'cavs_only'],
                       help='VSL enforcement mode')
    
    # Training configuration
    parser.add_argument('--timesteps', type=int, default=500_000,
                       help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel training environments')
    parser.add_argument('--eval-freq', type=int, default=25_000,
                       help='Evaluation frequency (steps per env)')
    parser.add_argument('--checkpoint-freq', type=int, default=25_000,
                       help='Checkpoint save frequency')
    
    # Hyperparameter options
    parser.add_argument('--use-optuna', action='store_true',
                       help='Use Optuna-tuned hyperparameters if available')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--buffer-size', type=int, default=None,
                       help='Override buffer size')
    
    # Execution options
    parser.add_argument('--parallel', action='store_true',
                       help='Run multiple configurations in parallel')
    parser.add_argument('--all-rewards', action='store_true',
                       help='Train all reward functions')
    parser.add_argument('--all-combinations', action='store_true',
                       help='Train all SAR combinations')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up temporary files after training')
    
    args = parser.parse_args()
    
    # Check SUMO
    if 'SUMO_HOME' not in os.environ:
        logger.error("Please set SUMO_HOME environment variable")
        sys.exit(1)
    
    # Prepare custom hyperparameters if specified
    custom_hyperparams = None
    if any([args.lr, args.batch_size, args.buffer_size]):
        custom_hyperparams = {}
        if args.lr:
            custom_hyperparams['learning_rate'] = get_linear_schedule(args.lr)
        if args.batch_size:
            custom_hyperparams['batch_size'] = args.batch_size
        if args.buffer_size:
            custom_hyperparams['buffer_size'] = args.buffer_size
    
    # Determine configurations to run
    if args.all_combinations:
        configurations = [
            ('full_metrics', 'absolute_speed', 'mobility'),
            ('full_metrics', 'absolute_speed', 'safety'),
            ('full_metrics', 'absolute_speed', 'balanced'),
            ('full_metrics', 'relative_speed', 'mobility'),
            ('full_metrics', 'relative_speed', 'safety'),
            ('full_metrics', 'relative_speed', 'balanced'),
            ('minimal', 'absolute_speed', 'mobility'),
            ('minimal', 'absolute_speed', 'safety'),
            ('minimal', 'absolute_speed', 'balanced'),
        ]
    elif args.all_rewards:
        configurations = [
            (args.state, args.action, 'mobility'),
            (args.state, args.action, 'safety'),
            (args.state, args.action, 'balanced'),
        ]
    else:
        configurations = [(args.state, args.action, args.reward)]
    
    # Run training
    if args.parallel and len(configurations) > 1:
        run_parallel_training(
            configurations,
            algorithm=args.algo,
            vsl_enforcement=args.vsl_mode,
            total_timesteps=args.timesteps,
            num_train_envs=args.n_envs,
            eval_freq=args.eval_freq,
            checkpoint_freq=args.checkpoint_freq,
            use_optuna_hyperparams=args.use_optuna,
            custom_hyperparams=custom_hyperparams
        )
    else:
        for state, action, reward in configurations:
            train_model(
                algorithm=args.algo,
                state_representation=state,
                action_strategy=action,
                reward_function=reward,
                vsl_enforcement=args.vsl_mode,
                total_timesteps=args.timesteps,
                num_train_envs=args.n_envs,
                eval_freq=args.eval_freq,
                checkpoint_freq=args.checkpoint_freq,
                use_optuna_hyperparams=args.use_optuna,
                custom_hyperparams=custom_hyperparams
            )
            
            if args.cleanup:
                model_name = f"{args.algo}_{reward}_{args.vsl_mode}"
                cleanup_temp_files(model_name)


if __name__ == "__main__":
    main()