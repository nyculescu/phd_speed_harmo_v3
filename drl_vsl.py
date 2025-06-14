import logging
import os
LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARN").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARN),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.info(f"Current working directory: {os.getcwd()}")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.getLogger('matplotlib').setLevel(logging.WARN) # Suppress matplotlib debug output
logging.getLogger('PIL').setLevel(logging.WARN) # Suppress PIL debug output
from stable_baselines3 import DQN
from typing import Optional
import torch.nn as nn
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
# from sb3_contrib import QRDQN
from flow_gen import *
from gymnasium.wrappers import TimeLimit
import gymnasium as gym
from datetime import datetime, timezone
import psutil
from time import sleep
import traci
from traci import FatalTraCIError, TraCIException
import subprocess
import sys
from pathlib import Path
from collections import deque
import pandas as pd
import time
import json
import copy
import multiprocessing as mp # Added for parallel processing
import csv
from tqdm import tqdm
import shutil
# from itertools import product # Added for generating combinations

""" SUMO configuration """
edges = ["seg_10_before","seg_9_before","seg_8_before","seg_7_before","seg_6_before","seg_5_before","seg_4_before","seg_3_before","seg_2_before","seg_1_before","seg_0_before","seg_0_after","seg_1_after"]
seg_10_before = ["seg_10_before_2", "seg_10_before_1", "seg_10_before_0"]
seg_9_before = ["seg_9_before_2", "seg_9_before_1", "seg_9_before_0"]
seg_8_before = ["seg_8_before_2", "seg_8_before_1", "seg_8_before_0"]
seg_7_before = ["seg_7_before_2", "seg_7_before_1", "seg_7_before_0"]
seg_6_before = ["seg_0_before_2", "seg_6_before_1", "seg_6_before_0"]
seg_5_before = ["seg_5_before_2", "seg_5_before_1", "seg_5_before_0"]
seg_4_before = ["seg_4_before_2", "seg_4_before_1", "seg_4_before_0"]
seg_3_before = ["seg_3_before_2", "seg_3_before_1", "seg_3_before_0"]
seg_2_before = ["seg_2_before_2", "seg_2_before_1", "seg_2_before_0"]
seg_1_before = ["seg_1_before_2", "seg_1_before_1", "seg_1_before_0"]
seg_0_before = ["seg_0_before_2", "seg_0_before_1", "seg_0_before_0"]
segments_before = [seg_10_before, seg_9_before, seg_8_before, seg_7_before, seg_6_before, seg_5_before, seg_4_before, seg_3_before, seg_2_before, seg_1_before, seg_0_before]
seg_0_after = ["seg_0_after_1", "seg_0_after_0"]
seg_1_after = ["seg_1_after_1", "seg_1_after_0"]
segments_after = [seg_0_after, seg_1_after]
loops_beforeA = ["loop_seg_0_before_2A", "loop_seg_0_before_1A", "loop_seg_0_before_0A"]
loops_beforeB = ["loop_seg_0_before_2B", "loop_seg_0_before_1B", "loop_seg_0_before_0B"]
loops_beforeC = ["loop_seg_0_before_2C", "loop_seg_0_before_1C", "loop_seg_0_before_0C"]
loops_beforeD = ["loop_seg_0_before_2D", "loop_seg_0_before_1D", "loop_seg_0_before_0D"]
loops_before = [loops_beforeA, loops_beforeB, loops_beforeC, loops_beforeD]
detectors_before = ["detector_seg_0_before_2", "detector_seg_0_before_1", "detector_seg_0_before_0"]
loops_after = ["loop_seg_0_after_1", "loop_seg_0_after_0"]
detectors_after = ["detector_seg_0_after_1", "detector_seg_0_after_0"]
detector_length = 50 # meters
BASE_TRAIN_SUMO_PORT = 8000
BASE_EVAL_SUMO_PORT = 9000
""" Curriculum learning for the DQN agent, which means gradually increasing the difficulty of the training scenarios. """
interval_length_h = 2 # hours
num_of_intervals = 10
num_test_envs_per_model = 1
num_train_envs_per_model = 1
num_envs_per_model = num_train_envs_per_model + num_test_envs_per_model
interval_length = 60 * interval_length_h
sumoExecutable_gui = 'sumo-gui.exe' if os.name == 'nt' else 'sumo-gui'
sumoExecutable_nogui = 'sumo.exe' if os.name == 'nt' else 'sumo'
SUMO_EXE_GUI = sumoExecutable_gui # NOTE: Change this to define which SUMO executable is used
sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', SUMO_EXE_GUI) # Default to GUI
MAX_OCCUPANCY = 100.0  # Occupancy percentage
MAX_FLOW = 10000.0    # vehicles/hour (theoretical maximum for 2.5 lanes)
MAX_SPEED_DIFF = 80.0  # km/h (130 - 50)
MAX_QUEUE_LENGTH_FOR_CRITICAL_SECTION = 575.0 * 3 / 7 # [vehicles], where 7 is a median vehicle length in meters, 3 is the no. of lanes and 575 is the total length in meters of 2 segments
OBSERVATION_SPACE_SIZE = 7
PROGRESS_BAR_ENABLED = True  # Enable progress bar for training
MAX_SPEED_MPS = 130 / 3.6       # 36.11 m/s approx
SPEED_TREND_CLIP = 1.0          # max absolute slope value for clipping
PORTS_PER_TUNING_PROCESS = 100 # Max trials * num_scenarios_per_trial + buffer

HYPER_PARAM_SIM_LENGTH = 3600
HYPER_PARAM_OPTUNA_STUD_TIMEOUT = 3600

OPTUNA_PARAMS_DIR = os.path.join("rl_models", "optuna_params")

BASE_DIR = Path(__file__).resolve().parent
TRAFFIC_ENV_SUMO_DIR = BASE_DIR / "traffic_environment" / "sumo"
SUMO_CONFIG_DIR = TRAFFIC_ENV_SUMO_DIR # Directory where .sumocfg files will be written
NORMALIZATION_BOUNDS_FILE = BASE_DIR / "rl_models" / "optuna_params" / "normalization_bounds.json"

# Enhanced hyperparameters based on traffic control research
ENHANCED_HYPERPARAMS = {
    "DQN": {
        # --- Q-Network Architecture ---
        "policy_kwargs": {
            "net_arch": [512, 256, 128], # Deeper network for complex traffic patterns
            "activation_fn": nn.ReLU
        },
        
        # --- Learning and Optimization ---
        "learning_rate": 1e-4,              # Slower, more stable learning rate
        "gamma": 0.995,                     # High discount factor for farsightedness
        "batch_size": 64,                   # Larger batch size for stable gradients
        "train_freq": (4, "step"),          # Update every 4 environment steps
        "gradient_steps": 1,                # 1 gradient step per update
        "tau": 1.0,                         # Hard target network update
        
        # --- Experience Replay and Exploration ---
        "buffer_size": 250000,              # Larger buffer for diverse traffic states
        "learning_starts": 10000,           # Delayed start for a quality initial buffer
        "exploration_fraction": 0.20,       # Longer exploration phase for traffic dynamics
        "exploration_initial_eps": 1.0,     # Start with full exploration
        "exploration_final_eps": 0.01,      # Lower final epsilon for more exploitation
        "target_update_interval": 10000     # Standard periodic target network updates
    }
}

SUMO_CFG_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
    <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
        <input>
            <net-file value="3_2_merge.net.xml"/>
            <route-files value="generated_flows_{file_postfix}_{index}.rou.xml"/>
            <additional-files value="loops_detectors.add.xml"/>
            <gui-settings-file value="colored.view.xml"/>
        </input>
        <processing>
            <lateral-resolution value="0.2"/>
        </processing>
    </configuration>
    """

def create_sumocfg(file_postfix, vsl_enforcement="recommend", model_idx_offset=0):
    output_dir = "./traffic_environment/sumo"
    os.makedirs(output_dir, exist_ok=True)

    # Generate configuration files
    for i in range(num_envs_per_model):
        # Ensure index is unique even if model_name is the same for SubprocVecEnv instances
        actual_index = i + model_idx_offset
        filename = f"3_2_merge_{file_postfix}_{actual_index}.sumocfg"
        filepath = os.path.join(output_dir, filename)
        
        # Format the template with current model and index
        content = SUMO_CFG_TEMPLATE.format(file_postfix=file_postfix, index=actual_index)
        
        # Write the content to the file
        with open(filepath, 'w') as file:
             file.write(content)
        
        logger.debug(f"Created {filepath}")

def train_env_constructor(idx, model_name, sim_length, num_of_episodes, reward_fn, vsl_enforcement="recommend", sumo_port_to_use=None, sumo_binary_to_use=None):
    def _init():
        port_for_env = sumo_port_to_use if sumo_port_to_use is not None else BASE_TRAIN_SUMO_PORT + idx
        
        # This call inside the constructor is missing sim_length
        env = Monitor(TrafficEnv(port=port_for_env,
                                model_name=model_name,
                                model_idx=idx,
                                sim_length=sim_length, # <-- This was the missing link
                                op_mode="train",
                                base_gen_car_distrib=["uniform", 2000],
                                num_of_episodes=num_of_episodes,
                                reward_fn=reward_fn,
                                vsl_enforcement=vsl_enforcement,
                                sumo_binary_path_override=sumo_binary_to_use))
        return env
    return _init

def eval_env_constructor(model_name, sim_length, reward_fn, vsl_enforcement="recommend", sumo_port_to_use=None, sumo_binary_to_use=None):
    def _init():
        port_for_eval_env = sumo_port_to_use if sumo_port_to_use is not None else BASE_EVAL_SUMO_PORT
        eval_model_idx = num_envs_per_model - 1

        env = Monitor(TimeLimit(TrafficEnv(port=port_for_eval_env,
                                            model_name=model_name,
                                            model_idx=eval_model_idx,
                                            sim_length=sim_length, # <-- PASS sim_length HERE
                                            op_mode="eval",
                                            base_gen_car_distrib=["uniform", 3000],
                                            num_of_episodes=1,
                                            reward_fn=reward_fn,
                                            vsl_enforcement=vsl_enforcement,
                                            sumo_binary_path_override=sumo_binary_to_use),
                                max_episode_steps=interval_length))
        return env
    return _init

def train_model(algorithm: str,
                reward_function: str = "balanced",
                num_of_episodes: int = 200, # Total episodes for the training run
                hyperparams: Optional[dict] = None,
                vsl_enforcement: str = "recommend",
                process_train_base_port: Optional[int] = None,
                process_eval_base_port: Optional[int] = None,
                sumo_binary_to_use: Optional[str] = None):
    """Trains a model using the specified algorithm and parameters."""

    TOTAL_TRAINING_TIMESTEPS = 100_000
    NO_OF_HR_OF_SIM = 4 # hours for training episodes
    EPISODE_SIM_LENGTH = 3600 * NO_OF_HR_OF_SIM
    NO_OF_HR_OF_EVAL = 1 # hours for evaluation
    EVAL_SIM_LENGTH = 3600 * NO_OF_HR_OF_EVAL
    EVAL_FREQ = 1800

    model_name = f"{algorithm}_{reward_function}_{vsl_enforcement}"
    log_dir = f"./logs/{model_name}/"
    model_dir = f"./rl_models/{model_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Determine base ports for SubprocVecEnv if provided
    # If num_train_envs_per_model > 1, each sub-env needs its own port.
    # The process_train_base_port is the starting port for this train_model call.
    current_train_base_port = process_train_base_port if process_train_base_port is not None else BASE_TRAIN_SUMO_PORT
    current_eval_base_port = process_eval_base_port if process_eval_base_port is not None else BASE_EVAL_SUMO_PORT

    train_env = SubprocVecEnv([
        train_env_constructor(i, model_name, EPISODE_SIM_LENGTH, num_of_episodes, reward_function, vsl_enforcement,
                              sumo_port_to_use=current_train_base_port + i,
                              sumo_binary_to_use=sumo_binary_to_use)
        for i in range(num_train_envs_per_model)
    ])

    env_eval = SubprocVecEnv([
        eval_env_constructor(model_name, EVAL_SIM_LENGTH, reward_function, vsl_enforcement,
                                                  sumo_port_to_use=current_eval_base_port,
                                                  sumo_binary_to_use=sumo_binary_to_use)])

    model = DQN("MlpPolicy", train_env, 
               learning_rate=hyperparams["learning_rate"],
               buffer_size=hyperparams["buffer_size"],
               batch_size=hyperparams["batch_size"],
               target_update_interval=hyperparams["target_update_interval"],
               exploration_fraction=hyperparams["exploration_fraction"],
               exploration_initial_eps=hyperparams["exploration_initial_eps"],
               exploration_final_eps=hyperparams["exploration_final_eps"],
               learning_starts=hyperparams["learning_starts"],
               train_freq=hyperparams["train_freq"],
               gradient_steps=hyperparams["gradient_steps"],
               tau=hyperparams["tau"],
               gamma=hyperparams["gamma"],
               policy_kwargs=hyperparams["policy_kwargs"],
               verbose=1, tensorboard_log=log_dir, device='cuda')

    logger.info(f"Training DQN with parameters: {hyperparams}")

    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    checkpoint_cb = CheckpointCallback(
        save_freq=EVAL_FREQ,
        save_path=model_dir,
        name_prefix=f"rl_model_{model_name}",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1
    )

    no_improve_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=1, # Allow more evaluations without improvement
        min_evals=1,
        verbose=1
    )

    eval_cb = EvalCallback(
        env_eval,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        callback_after_eval=no_improve_cb,
        verbose=1
    )

    custom_cb = CustomMetricsCallback()

    # Training loop
    try:
        model.learn(total_timesteps=TOTAL_TRAINING_TIMESTEPS,
                    callback=[checkpoint_cb, eval_cb, custom_cb],
                    progress_bar=PROGRESS_BAR_ENABLED,
                    reset_num_timesteps=False)
        model.save(os.path.abspath(f"./rl_models/{model_name}/{model_name}_last.zip"))
    except KeyboardInterrupt:
        logger.warning(f"Training for {model_name} interrupted by user.")
    except Exception as e:
        logger.error(f"Error during training for {model_name}: {e}")
    finally:
        train_env.close()
        env_eval.close()
        logger.info(f"Finished training for {model_name}")

def test_model(algorithm, reward_function, vsl_enforcement="recommend"):
    """Test a trained DQN model with comprehensive evaluation."""
    model_name = f"{algorithm}_{reward_function}_{vsl_enforcement}"
    model_load_path = Path(f"rl_models/{model_name}/best_model.zip")
    if not model_load_path.exists():
        model_load_path = Path(f"rl_models/{model_name}/{model_name}_last.zip") # Try the final saved model
        if not model_load_path.exists():
            logger.error(f"Could not find model for {model_name} at {model_load_path} or best_model.zip. Exiting test.")
            return
    
    try:
        model = DQN.load(str(model_load_path))
        logger.info(f"Loaded model from {model_load_path} for testing.")
    except Exception as e:
        logger.error(f"Error loading model from {model_load_path}: {e}. Exiting test.")
        return

    NO_OF_HR_OF_TEST = 4 # hours for testing
    TEST_SIM_LENGTH = 3600 * NO_OF_HR_OF_TEST

    env = TrafficEnv(port=BASE_EVAL_SUMO_PORT,
                     model_name=model_name,
                     model_idx=0,
                     sim_length=TEST_SIM_LENGTH,
                     base_gen_car_distrib=["bimodal", 3],
                     num_of_episodes=1,
                     reward_fn=reward_function,
                     vsl_enforcement=vsl_enforcement,
                     sumo_binary_path_override=sumoBinary) # Use the globally defined sumoBinary

    obs, info = env.reset()
    total_reward = 0
    step_count = 0

    expected_test_steps = env.sim_length // env.aggregation_time

    logger.info(f"Starting test for {model_name}. Expected steps: {expected_test_steps}")

    with tqdm(total=expected_test_steps, desc=f"Testing {model_name}", unit="step") as pbar:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            pbar.update(1)
            pbar.set_postfix_str(f"Reward: {reward:.2f}, Total Reward: {total_reward:.2f}")
            if done or truncated:
                logger.info(f"Test episode finished. Done: {done}, Truncated: {truncated}")
                break
    
    pbar.close() # Ensure progress bar is closed
    final_summary = env.logger.get_summary_statistics()
    env.close()

    print(f"\n--- Comprehensive Test Results for {model_name} ---")
    # Pretty print the dictionary
    for key, value in final_summary.items():
        print(f"{key:<30}: {value}")
    
    log_filename = f"test_run_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    env.logger.save_to_csv(filename=log_filename)
    print(f"\nDetailed step-by-step log saved to: ./logs/traffic_data/{log_filename}")

    print(f"\nTest Results for {model_name}:")
    print(f"Total Steps: {step_count}")
    print(f"Total Reward: {total_reward:.2f}")
    if step_count > 0:
        print(f"Average Reward per Step: {total_reward/step_count:.3f}")
    
    # Attempt to get more detailed final metrics if available in info
    final_flow_downstream = info.get('flow_downstream', 'N/A')
    final_avg_speed = info.get('avg_speed_before', 'N/A') # Assuming avg_speed_before is relevant
    final_collisions = info.get('collisions', 'N/A')
    
    print(f"Final Flow Rate (Downstream): {final_flow_downstream} veh/h")
    print(f"Final Average Speed (Upstream of VSL): {final_avg_speed} m/s")
    print(f"Total Collisions during test: {final_collisions}")
    logger.info(f"Test for {model_name} complete. Total steps: {step_count}, Total reward: {total_reward:.2f}")

def run_training_for_combination(config_tuple):
    reward_fn, vsl_mode, process_id, algo_used, parallel_sumo_binary, use_hyperparams_by_optuna = config_tuple
    
    base_port_for_this_process = BASE_TRAIN_SUMO_PORT + process_id * 10
    main_train_base_port = base_port_for_this_process
    main_eval_base_port = base_port_for_this_process + num_train_envs_per_model + 5

    config_model_name_for_training = f"{algo_used}_{reward_fn}_{vsl_mode}"
    logger.info(f"Process {process_id}: Starting combination {config_model_name_for_training}. Train Port Base: {main_train_base_port}, Eval Port Base: {main_eval_base_port}")

    # --- Hyperparameter Loading and Processing ---
    # 1. Start with a base: deep copy of ENHANCED_HYPERPARAMS or a basic default
    final_hyperparams = None
    if algo_used in ENHANCED_HYPERPARAMS:
        final_hyperparams = copy.deepcopy(ENHANCED_HYPERPARAMS[algo_used])
        logger.debug(f"Process {process_id}: Initialized with ENHANCED_HYPERPARAMS for {algo_used}.")
    else:
        logger.error(f"Process {process_id}: Algorithm {algo_used} not found in ENHANCED_HYPERPARAMS. Using basic DQN defaults.")
        final_hyperparams = {
            "learning_rate": 1e-4, "buffer_size": 100000, "batch_size": 128,
            "target_update_interval": 1000, "exploration_fraction": 0.1,
            "exploration_initial_eps": 1.0, "exploration_final_eps": 0.01,
            "learning_starts": 1000, "train_freq": (4, "step"), "gradient_steps": 1,
            "tau": 1.0, "gamma": 0.99,
            "policy_kwargs": {"net_arch": [256, 256, 128], "activation_fn": nn.ReLU}
        }

    if use_hyperparams_by_optuna:
        optuna_params_dir = os.path.join("rl_models", "optuna_params")
        specific_optuna_params_filename = f"best_optuna_params_{algo_used}_{reward_fn}_{vsl_mode}.json"
        specific_optuna_params_path = os.path.join(optuna_params_dir, specific_optuna_params_filename)

        if os.path.exists(specific_optuna_params_path):
            try:
                with open(specific_optuna_params_path, "r") as f:
                    loaded_json_params = json.load(f)
                logger.info(f"Process {process_id}: Successfully loaded Optuna params from {specific_optuna_params_path}")

                # Determine if the loaded params are nested (e.g., {"DQN": {...}}) or flat
                optuna_params_to_merge = {}
                if algo_used in loaded_json_params and isinstance(loaded_json_params[algo_used], dict):
                    optuna_params_to_merge = loaded_json_params[algo_used]
                else: # Assume it's a flat dictionary of parameters
                    optuna_params_to_merge = loaded_json_params
                
                # Merge Optuna params into final_hyperparams. Optuna values will override base/enhanced values.
                for key, value in optuna_params_to_merge.items():
                    if key == "policy_kwargs" and isinstance(value, dict) and \
                       "policy_kwargs" in final_hyperparams and isinstance(final_hyperparams["policy_kwargs"], dict):
                        # Deep merge for policy_kwargs
                        for pk_key, pk_value in value.items():
                            final_hyperparams["policy_kwargs"][pk_key] = pk_value
                    else:
                        final_hyperparams[key] = value
                
                logger.info(f"Process {process_id}: Merged Optuna params into the hyperparameter set.")

            except Exception as e:
                logger.warning(f"Process {process_id}: Could not load/parse Optuna params from {specific_optuna_params_path}: {e}. Will use base/enhanced defaults.")
        else:
            logger.warning(f"Process {process_id}: Optuna params file not found at {specific_optuna_params_path}. Will use base/enhanced defaults.")
    else:
        logger.info(f"Process {process_id}: Not configured to use Optuna params. Will use base/enhanced defaults.")

    # --- Post-merge/Post-load Processing for final_hyperparams ---
    # Ensure policy_kwargs dictionary exists
    if "policy_kwargs" not in final_hyperparams or not isinstance(final_hyperparams["policy_kwargs"], dict):
        final_hyperparams["policy_kwargs"] = {}

    # Handle "net_arch_str" (could be at top level from flat Optuna file or inside policy_kwargs)
    net_arch_source_str = None
    if "net_arch_str" in final_hyperparams:
        net_arch_source_str = final_hyperparams.pop("net_arch_str")
    elif "net_arch_str" in final_hyperparams["policy_kwargs"]:
        net_arch_source_str = final_hyperparams["policy_kwargs"].pop("net_arch_str")
    
    if net_arch_source_str:
        try:
            final_hyperparams["policy_kwargs"]["net_arch"] = [int(x.strip()) for x in net_arch_source_str.split(',')]
            logger.debug(f"Process {process_id}: Parsed net_arch_str '{net_arch_source_str}' to {final_hyperparams['policy_kwargs']['net_arch']}")
        except ValueError as e:
            logger.warning(f"Process {process_id}: Could not parse net_arch_str '{net_arch_source_str}': {e}. Ensuring default net_arch.")
            if "net_arch" not in final_hyperparams["policy_kwargs"]: # If parsing failed and no net_arch exists
                 final_hyperparams["policy_kwargs"]["net_arch"] = [256, 256, 128] # Fallback

    # Ensure 'net_arch' and 'activation_fn' are in policy_kwargs with correct types
    if "net_arch" not in final_hyperparams["policy_kwargs"]:
        final_hyperparams["policy_kwargs"]["net_arch"] = [256, 256, 128] # Default
        logger.debug(f"Process {process_id}: 'net_arch' not found in policy_kwargs, set to default.")
        
    if isinstance(final_hyperparams["policy_kwargs"].get("activation_fn"), str):
        if final_hyperparams["policy_kwargs"]["activation_fn"] == "nn.ReLU":
            final_hyperparams["policy_kwargs"]["activation_fn"] = nn.ReLU
            logger.debug(f"Process {process_id}: Converted 'activation_fn' string to nn.ReLU object.")
        else: # Unknown string, fallback or log error
            logger.warning(f"Process {process_id}: Unknown string for activation_fn: {final_hyperparams['policy_kwargs']['activation_fn']}. Setting to nn.ReLU.")
            final_hyperparams["policy_kwargs"]["activation_fn"] = nn.ReLU
    elif "activation_fn" not in final_hyperparams["policy_kwargs"]:
        final_hyperparams["policy_kwargs"]["activation_fn"] = nn.ReLU # Default
        logger.debug(f"Process {process_id}: 'activation_fn' not found in policy_kwargs, set to nn.ReLU default.")

    # Convert "train_freq" list/int to tuple
    if "train_freq" in final_hyperparams:
        if isinstance(final_hyperparams["train_freq"], list):
            final_hyperparams["train_freq"] = tuple(final_hyperparams["train_freq"])
            logger.debug(f"Process {process_id}: Converted 'train_freq' list to tuple: {final_hyperparams['train_freq']}")
        elif isinstance(final_hyperparams["train_freq"], int):
            final_hyperparams["train_freq"] = (final_hyperparams["train_freq"], "step")
            logger.debug(f"Process {process_id}: Converted 'train_freq' int to tuple: {final_hyperparams['train_freq']}")
    # --- End Hyperparameter Loading and Processing ---

    try:
        logger.info(f"Process {process_id}: Creating SUMO config for {config_model_name_for_training}...")
        create_sumocfg(config_model_name_for_training, vsl_mode) 
                                           
        logger.info(f"Process {process_id}: Training model {config_model_name_for_training} with final hyperparams: {final_hyperparams}")
        train_model(algorithm=algo_used,
                    reward_function=reward_fn,
                    hyperparams=final_hyperparams, # Pass the processed hyperparams
                    vsl_enforcement=vsl_mode,
                    process_train_base_port=main_train_base_port,
                    process_eval_base_port=main_eval_base_port,
                    sumo_binary_to_use=parallel_sumo_binary)
        
        logger.info(f"Process {process_id}: Successfully completed training for {config_model_name_for_training}")
        return f"Success: {config_model_name_for_training}"
    except Exception as e:
        logger.error(f"Process {process_id}: FAILED for {config_model_name_for_training}. Error: {e}", exc_info=True)
        return f"Failure: {config_model_name_for_training} - {e}"

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Classes """
class TrafficEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, port, model_name, model_idx, sim_length, base_gen_car_distrib, 
                 num_of_episodes, 
                 op_mode: str = "train", # FIXME: remove it
                 reward_fn="balanced", 
                 vsl_enforcement: str = "recommend",
                 sumo_binary_path_override: Optional[str] = None, 
                 normalization_bounds_path: Optional[str] = None):
        super(TrafficEnv, self).__init__()
        self.default_speed_limit = 130
        self.port = port
        self.sim_length = sim_length
        self.sumo_step_length = 1  # [s] SUMO step length
        self.model_name = model_name
        self.model_idx = model_idx
        self.effective_model_name_for_files = model_name
        self.effective_model_idx_for_files = model_idx
        self.sumo_binary_path_override = sumo_binary_path_override
        self.skip_flow_generation = False
        self.aggregation_time = 60  # [s] Data aggregation duration
        self.sumo_process = None
        self.sumo_max_retries = 3
        self.is_sumo_initialized = False  # Track SUMO initialization state
        self.collisions = []
        self.collisions_penalty = 0
        self.gen_car_distrib = base_gen_car_distrib
        self.logger = TrafficDataLogger(model_name=model_name, log_dir=Path(f"./logs/{model_name}_{reward_fn}_{vsl_enforcement}"))
        self.num_of_episodes = num_of_episodes
        self.reward_fn = reward_fn
        
        # Historical data for analysis
        self.flow_downstream_history = deque(maxlen=5)
        self.occupancy_downstream_history = deque(maxlen=5)
        self.speed_history = deque(maxlen=15)
        self.flow_smoothed = 0.0
        self.occupancy_smoothed = 0.0
        
        # Action and observation spaces
        self._load_or_set_normalization_bounds(normalization_bounds_path)

        self.action_space = gym.spaces.Discrete(5)
        self.current_speed_limit = self.default_speed_limit
        
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0, -np.inf, 0, 50]),
            high=np.array([
                self.default_speed_limit/3.6,
                np.inf, np.inf, np.inf, np.inf, 1.0, 130
            ]),
            shape=(OBSERVATION_SPACE_SIZE,),
            dtype=np.float64
        )
        
        # State variables
        self.queue_length_upstream = 0
        self.flow_upstream = 0
        self.flow_downstream = 0
        self.avg_speed_before = 0
        self.occupancy_upstream = 0
        self.simulation_step = 0

        # Track a moving average of recent rewards. If this average falls below a threshold for a certain number of steps, terminate the episode early.
        self.reward_window = deque(maxlen=50)  # Track last 50 rewards
        self.reward_threshold = -5 # Threshold for early termination in tuning

        # VSL Enforcement Mode Configuration
        # Options: "all_vehicles", "electric_only", "recommend"
        self.vsl_enforcement = vsl_enforcement
        
        # Aattributes for start_sumo customization
        self._sumo_start_context_prefix = ""  # e.g., "Tuning " for the child class
        self._default_sumo_binary_for_env = sumoBinary # Default for TrafficEnv
        self._sumo_retry_sleep_func = lambda attempt, max_retries: max_retries + attempt # Default retry logic

        self.veh_passed_downstream = 0  # FIXME: Temp debug

        self.training_scenarios = [
            {"demand": 2500, "pattern": "uniform"},
            {"demand": 3000, "pattern": "uniform"}, 
            {"demand": 3500, "pattern": "uniform"},
            {"demand": 4000, "pattern": "uniform"},
            {"demand": 4500, "pattern": "uniform"},
            {"demand": 5000, "pattern": "uniform"},
        ]
        self.gen_car_distrib = base_gen_car_distrib 

    def _load_or_set_normalization_bounds(self, bounds_path: Optional[str]):
        """Loads normalization bounds from a file or falls back to hardcoded defaults."""
        if bounds_path and os.path.exists(bounds_path):
            try:
                with open(bounds_path, 'r') as f:
                    bounds = json.load(f)
                self.max_flow = bounds.get("max_flow", MAX_FLOW)
                self.max_occupancy = bounds.get("max_occupancy", MAX_OCCUPANCY)
                self.max_queue_length = bounds.get("max_queue_length", MAX_QUEUE_LENGTH_FOR_CRITICAL_SECTION)
                logger.info(f"Port {self.port}: Successfully loaded dynamic normalization bounds from {bounds_path}.")
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Port {self.port}: Failed to read bounds from {bounds_path}, using defaults. Error: {e}")
                self._set_default_normalization_bounds()
        else:
            if bounds_path: # Path was given but not found
                logger.warning(f"Port {self.port}: Bounds file not found at {bounds_path}, using defaults.")
            else: # Path was not given
                logger.debug(f"Port {self.port}: No bounds file path provided, using default normalization bounds.")
            self._set_default_normalization_bounds()
    
    def _set_default_normalization_bounds(self):
        """Sets the hardcoded default normalization bounds as instance variables."""
        self.max_flow = MAX_FLOW
        self.max_occupancy = MAX_OCCUPANCY
        self.max_queue_length = MAX_QUEUE_LENGTH_FOR_CRITICAL_SECTION

    def _get_sumo_log_identifier(self):
        """Helper to get a consistent identifier for SUMO instance logging."""
        # For TrafficEnvForTuning, effective_model_name_for_files includes "_tune_"
        # and effective_model_idx_for_files is the scenario_id.
        # For TrafficEnv, these are the main model name and sub-env index.
        return f"{self._sumo_start_context_prefix}{self.effective_model_name_for_files}_{self.effective_model_idx_for_files}"
    
    def _ensure_clean_traci_state(self):
        """Ensure TraCI is in a clean state before starting SUMO."""
        try:
            if traci.isLoaded():
                logger.debug("TraCI connection found active, closing it...")
                traci.close()
        except Exception as e:
            logger.debug(f"Error while checking/closing TraCI: {e}")
        
        # Small delay to ensure connection is fully closed
        time.sleep(0.1)

    def _start_sumo(self):
        """Initialize SUMO simulation - only start if not already running properly."""
        log_id = self._get_sumo_log_identifier()

        if self.is_sumo_initialized and self.sumo_process and psutil.pid_exists(self.sumo_process.pid):
            try:
                traci.simulation.getTime()
                logger.debug(f"SUMO ({log_id}) is already running and responsive.")
                return
            except (FatalTraCIError, TraCIException, ConnectionResetError, BrokenPipeError):
                logger.warning(f"SUMO process ({log_id}) exists but not responsive, restarting...")
                self.is_sumo_initialized = False # Mark for restart
        
        if self.sumo_process and psutil.pid_exists(self.sumo_process.pid):
            self._close_sumo(f"Restarting SUMO for initialization ({log_id})")
            sleep(3) # Give a bit more time for resources to free up
        elif self.sumo_process and not psutil.pid_exists(self.sumo_process.pid):
            logger.debug(f"SUMO process handle existed for {log_id} but PID was not found. Clearing handle.")
            self.sumo_process = None # Clear stale handle

        self._ensure_clean_traci_state()
        
        for attempt in range(self.sumo_max_retries):
            try:
                port = self.port
                
                route_file = f"./traffic_environment/sumo/generated_flows_{self.effective_model_name_for_files}_{self.effective_model_idx_for_files}.rou.xml"
                if not os.path.exists(route_file) or os.path.getsize(route_file) == 0:
                    logger.error(f"Route file missing or empty: {route_file} on attempt {attempt + 1} for {log_id}.")

                current_sumo_binary = self.sumo_binary_path_override if self.sumo_binary_path_override else self._default_sumo_binary_for_env
                
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                sumo_log_file = f"./logs/sumo_log/{self.effective_model_name_for_files}_{self.effective_model_idx_for_files}_{timestamp_str}.txt"

                sumo_cmd = [
                    current_sumo_binary, "-c",
                    f"./traffic_environment/sumo/3_2_merge_{self.effective_model_name_for_files}_{self.effective_model_idx_for_files}.sumocfg",
                    '--start',
                    "--default.emergencydecel=7",
                    '--random-depart-offset=3600',
                    "--remote-port", str(port),
                    f"--step-length={self.sumo_step_length}",
                    "--default.action-step-length=0.2",
                    f"--end={self.sim_length}",
                    # "--quit-on-end",
                    "--no-step-log", 
                    "--no-warnings",
                    "--log", sumo_log_file
                ]

                self.sumo_process = subprocess.Popen(sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # Give SUMO a moment to launch and potentially fail
                time.sleep(0.2) 
                # poll() returns the exit code if the process has terminated, or None otherwise.
                exit_code = self.sumo_process.poll()
                if exit_code is not None:
                    # Process terminated immediately. Capture and log the error.
                    out, err = self.sumo_process.communicate()
                    logger.error(f"SUMO process failed on launch with exit code {exit_code}.")
                    logger.error(f"SUMO stdout: {out.decode()}")
                    logger.error(f"SUMO stderr: {err.decode()}")
                    # Raise an exception to stop the attempt.
                    raise RuntimeError("SUMO failed to start. Check logs for details.")

                logger.debug(f"SUMO command: {' '.join(sumo_cmd)}")
                logger.debug(f"Expected simulation end time: {self.sim_length}s")

                logger.info(f"Attempting to connect to SUMO ({log_id}) on port {port}")
                time.sleep(0.5) 
                try:
                    traci.init(port=port, numRetries=5)
                except Exception as e:
                    if self.sumo_process:
                        out, err = self.sumo_process.communicate(timeout=2)
                        logger.error(f"SUMO stderr: {err.decode()}")
                    raise e
                logger.info(f"Successfully connected to SUMO ({log_id}) on port {port} for {self.sim_length}s simulation")
                self.is_sumo_initialized = True
                break
                
            except (FatalTraCIError, TraCIException, ConnectionRefusedError) as e:
                logger.error(f"Attempt {attempt + 1} to start/connect SUMO ({log_id}) failed: {e}")
                self._close_sumo(f"Failed to start/connect SUMO ({log_id}) on attempt {attempt+1}")
                if attempt < self.sumo_max_retries - 1:
                    sleep(self._sumo_retry_sleep_func(attempt, self.sumo_max_retries)) # Use customized retry sleep
                else:
                    logger.error(f"Max retries reached for starting SUMO ({log_id}). Raising exception.")
                    raise e

    def step(self, action: int):
        """Execute one step in the environment."""
        # Initialize SUMO if not already done
        if not self.is_sumo_initialized:
            self._start_sumo()
        
        # Check SUMO responsiveness
        try:
            current_time = traci.simulation.getTime()
        except (FatalTraCIError, TraCIException):
            logger.error("Lost connection to SUMO, restarting...")
            self.is_sumo_initialized = False
            self._start_sumo()
            current_time = traci.simulation.getTime()
        
        # Apply action: gradual speed limit changes
        speed_changes = [-10, -5, 0, +5, +10]  # Larger action space
        previous_speed_limit = self.current_speed_limit
        proposed_speed_limit = self.current_speed_limit + speed_changes[action]
        
        # Invalid action penalty and clamping
        invalid_action_penalty = 0
        
        # Safety constraint: limit consecutive changes
        if hasattr(self, 'recent_changes') and len(self.recent_changes) >= 3:
            if all(abs(change) >= 5 for change in list(self.recent_changes)[-3:]):
                # Prevent excessive consecutive changes
                proposed_speed_limit = previous_speed_limit
                invalid_action_penalty = -2.0
        
        # Comfort constraint: maximum 20 km/h change as per literature
        max_change = 20
        if abs(proposed_speed_limit - previous_speed_limit) > max_change:
            proposed_speed_limit = previous_speed_limit + np.sign(proposed_speed_limit - previous_speed_limit) * max_change
            invalid_action_penalty = -1.0
        
        # Apply bounds
        self.current_speed_limit = max(50, min(130, proposed_speed_limit))
        
        # Track recent changes
        if not hasattr(self, 'recent_changes'):
            self.recent_changes = deque(maxlen=5)
        self.recent_changes.append(self.current_speed_limit - previous_speed_limit)
        
        # Apply VSL enforcement using the new method
        self._apply_vsl_enforcement(self.current_speed_limit)
        
        # Initialize data collection variables
        flow_upstream_temp = 0
        flow_downstream_temp = 0
        queue_length_temp = 0
        mean_speeds_downstream = 0
        mean_speeds_upstream = 0
        occupancy_upstream_temp = 0
        
        # Simulation steps and data aggregation
        num_sumo_steps = int(self.aggregation_time / self.sumo_step_length)
        for step in range(num_sumo_steps):
            try:
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                self.simulation_step += 1
            except (FatalTraCIError, TraCIException):
                logger.error("Lost connection during simulation steps. Terminating episode.")
                # Return a valid 5-tuple to signal a terminal state
                # Get the last valid observation before the crash
                last_observation = self._preprocess_state(np.array([
                    self.avg_speed_before, self.flow_upstream, self.flow_smoothed,
                    self.queue_length_upstream, self._calculate_speed_trend(),
                    self.occupancy_smoothed / 100.0, self.current_speed_limit
                ], dtype=np.float64))
                
                # Return a terminal observation with a large negative reward
                return last_observation, -10.0, True, False, {} 
            
            # Collect traffic measurements
            flow_upstream_temp += traci.edge.getLastStepVehicleNumber("seg_0_before")
            flow_downstream_temp += traci.edge.getLastStepVehicleNumber("seg_0_after")
            
            """
            # Measure queue for all upstream segments based on halting vehicles
            queue_length_temp += sum(
                traci.lane.getLastStepHaltingNumber(lane_id) * 7.5
                for segment in segments_before  # Use your defined segments_before list
                for lane_id in segment
            )
            """
            
            # Measure queue for specific critical segments (e.g., near merge point)
            critical_segments = [seg_1_before, seg_0_before]
            queue_length_temp += sum(
                traci.lane.getLastStepHaltingNumber(lane_id) * 7.5
                for segment in critical_segments
                for lane_id in segment
            )

            # Speed measurements
            mean_speeds_downstream += traci.edge.getLastStepMeanSpeed("seg_0_before")
            mean_speeds_upstream += traci.edge.getLastStepMeanSpeed("seg_0_after")
            
            # Occupancy from induction loops
            occupancy_upstream_temp += sum([
                traci.inductionloop.getLastStepOccupancy(loop_id)
                for loop_ids in loops_before
                for loop_id in loop_ids
            ]) / len([loop for loops in loops_before for loop in loops])
            
            # Collision detection
            collisions_in_step = traci.simulation.getCollidingVehiclesNumber()
            if collisions_in_step > 0:
                self.collisions.append(current_time)
        
        # Process collected data
        self.avg_speed_before = mean_speeds_downstream / self.aggregation_time
        self.flow_upstream = (flow_upstream_temp / self.aggregation_time) * 3600
        self.flow_downstream = (flow_downstream_temp / self.aggregation_time) * 3600
        self.queue_length_upstream = queue_length_temp / self.aggregation_time
        self.occupancy_upstream = min(occupancy_upstream_temp / self.aggregation_time, 100.0)

        # Update historical data for smoothing
        self.flow_downstream_history.append(self.flow_downstream)
        self.occupancy_downstream_history.append(self.occupancy_upstream)
        self.speed_history.append(self.avg_speed_before)
        
        # Calculate smoothed values
        self.flow_smoothed = np.mean(list(self.flow_downstream_history)) if self.flow_downstream_history else 0
        self.occupancy_smoothed = np.mean(list(self.occupancy_downstream_history)) if self.occupancy_downstream_history else 0
        
        # Collision penalty (2-hour sliding window)
        expiration_time = current_time - (2 * 3600)
        self.collisions = [t for t in self.collisions if t > expiration_time]
        self.collisions_penalty = -5 if len(self.collisions) > 2 else 0
        
        # Calculate reward
        reward = self._calculate_reward(invalid_action_penalty)

        self.reward_window.append(reward)
        
        # Prepare observation
        speed_trend_val = self._calculate_speed_trend()
        raw_observation = np.array([
            self.avg_speed_before,
            self.flow_upstream,
            self.flow_smoothed,
            self.queue_length_upstream,
            speed_trend_val,
            self.occupancy_smoothed / 100.0,
            self.current_speed_limit
        ], dtype=np.float64)

        # Normalize observation for DQN
        observation = self._preprocess_state(raw_observation)
                
        # Check termination conditions
        # End when simulation time reaches limit OR no more vehicles expected
        done = (current_time >= self.sim_length) or (traci.simulation.getMinExpectedNumber() <= 0) or (len(self.reward_window) == self.reward_window.maxlen and np.mean(self.reward_window) < self.reward_threshold)
        
        # Log data
        self.logger.log_step_data(
            current_time, self.current_speed_limit, self.flow_upstream,
            self.flow_downstream, self.occupancy_upstream, self.queue_length_upstream,
            reward, action, self.avg_speed_before
        )
        
        info = {
            'flow_upstream': self.flow_upstream,
            'flow_downstream': self.flow_downstream,
            'avg_speed_before': self.avg_speed_before, # <-- Add this line
            'occupancy': self.occupancy_upstream,
            'queue_length_upstream': self.queue_length_upstream, # <-- Add this line
            'speed_limit': self.current_speed_limit,
            'collisions': len(self.collisions),
            'simulation_time': current_time,
            'simulation_step': self.simulation_step
        }
        
        self.veh_passed_downstream += flow_downstream_temp # FIXME: Temp debug
        logger.debug(f"No. of vehicles arrived: {self.veh_passed_downstream}") # FIXME: Temp debug

        return observation, reward, done, False, info

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Close existing SUMO if running
        if self.is_sumo_initialized:
            self._close_sumo("Environment reset")
        
        if not self.skip_flow_generation:
            # Randomly select a scenario for the new episode
            scenario = np.random.choice(self.training_scenarios)
            self.gen_car_distrib = [scenario["pattern"], scenario["demand"]]
            logger.info(f"Resetting env. New scenario: Demand={self.gen_car_distrib[1]} veh/hr")

            # Generate the flow file for this specific scenario
            if self.gen_car_distrib[0] == 'uniform':
                flow_generation_fix_num_veh(
                    self.effective_model_name_for_files, 
                    self.effective_model_idx_for_files,
                    self.gen_car_distrib[1],
                    self.sim_length,
                    1, # Each episode is now self-contained
                    1
                )
            elif self.gen_car_distrib[0] == 'bimodal':
                flow_generation(
                    self.effective_model_name_for_files, 
                    self.effective_model_idx_for_files,
                    bimodal_distribution_24h(self.gen_car_distrib[1]), 
                    self.sim_length
                )

        # Reset state variables
        self.current_speed_limit = self.default_speed_limit
        self.flow_upstream = 0
        self.flow_downstream = 0
        self.queue_length_upstream = 0
        self.occupancy_upstream = 0
        self.avg_speed_before = 0
        self.collisions = []
        self.collisions_penalty = 0
        self.simulation_step = 0
        self.is_sumo_initialized = False
        
        # Clear historical data
        self.flow_downstream_history.clear()
        self.occupancy_downstream_history.clear()
        self.speed_history.clear()
        
        # Start fresh SUMO instance
        self._start_sumo()
                
        raw_observation = np.array([
            self.default_speed_limit / 3.6,
            0.0, 0.0, 0.0, 0.0, 0.0,
            self.default_speed_limit
        ], dtype=np.float64)

        self.veh_passed_downstream = 0  # FIXME: Temp debug

        observation = self._preprocess_state(raw_observation)

        info = {
            'flow_upstream': 0, 'flow_downstream': 0, 'occupancy': 0,
            'queue_length': 0, 'speed_limit': self.default_speed_limit,
            'collisions': 0, 'simulation_time': 0, 'simulation_step': 0
        }
        
        return observation, info

    def _calculate_reward(self, invalid_action_penalty):
        """
        Calculate reward based on selected reward function.
        Implements multi-objective reward functions from recent research.
        """
        if self.reward_fn == "mobility":
            return self._reward_mobility_focused(invalid_action_penalty)
        elif self.reward_fn == "safety":
            return self._reward_safety_focused(invalid_action_penalty)
        elif self.reward_fn == "balanced":
            return self._reward_balanced(invalid_action_penalty)
        else:
            return self._reward_balanced(invalid_action_penalty)  # Default

    def _reward_mobility_focused(self, invalid_action_penalty):
        """Enhanced mobility reward incorporating capacity utilization metrics."""
        # Flow efficiency with capacity consideration
        capacity_utilization = min(self.flow_smoothed / MAX_FLOW, 1.0)
        R_flow = capacity_utilization * 0.5
        
        # Throughput reward (vehicles processed per hour)
        throughput_reward = min(self.flow_downstream / MAX_FLOW, 1.0) * 0.2
        
        # Speed harmonization (reduce variance)
        R_smooth = self._calculate_speed_smoothness() * 0.2
        
        # Queue penalty with exponential scaling
        queue_penalty = min((self.queue_length_upstream / MAX_QUEUE_LENGTH_FOR_CRITICAL_SECTION)**2, 1.0) * 0.1
        
        return R_flow + throughput_reward + R_smooth - queue_penalty + invalid_action_penalty + self.collisions_penalty

    def _reward_safety_focused(self, invalid_action_penalty):
        """
        Safety-focused reward function emphasizing crash risk reduction and speed variance.
        Targets 19.4% lower crash risk as shown in research.
        """
        # Primary: Speed harmonization (reduce variance)
        R_smooth = self._calculate_speed_smoothness() * 0.4
        
        # Secondary: Average speed maintenance
        avg_speed_reward = min(self.avg_speed_before / (self.default_speed_limit / 3.6), 1.0) * 0.3
        
        # Tertiary: Flow efficiency
        R_flow = min(self.flow_smoothed / MAX_FLOW, 1.0) * 0.2
        
        # Enhanced collision penalty
        collision_penalty = self.collisions_penalty * 2  # Double weight for safety focus
        
        reward = R_smooth + avg_speed_reward + R_flow + collision_penalty + invalid_action_penalty
        return float(reward)

    def _reward_balanced(self, invalid_action_penalty):   
        # Base components (your existing approach)
        R_flow = min(self.flow_smoothed / MAX_FLOW, 1.0) * 0.25
        
        # Enhanced safety component (higher weight based on literature)
        speed_variance = np.var(list(self.speed_history)) if len(self.speed_history) > 2 else 0.0
        R_safety = max(0.0, 1.0 - (speed_variance / 400.0)) * 0.35  # Increased weight
        
        # Control smoothness (literature emphasizes this)
        speed_change_magnitude = abs(self.current_speed_limit - getattr(self, 'previous_speed_limit', self.current_speed_limit))
        R_smoothness = max(0.0, 1.0 - (speed_change_magnitude / 20.0)) * 0.15
        
        # Efficiency with target consideration
        target_speed = 100.0 / 3.6  # 100 km/h optimal
        speed_efficiency = 1.0 - abs(self.avg_speed_before - target_speed) / target_speed
        R_efficiency = max(0.0, speed_efficiency) * 0.15
        
        # Queue prevention (exponential penalty)
        queue_penalty = min((self.queue_length_upstream / MAX_QUEUE_LENGTH_FOR_CRITICAL_SECTION)**1.5, 1.0) * 0.1
        
        total_reward = R_flow + R_safety + R_smoothness + R_efficiency - queue_penalty + invalid_action_penalty + self.collisions_penalty
        
        # Track previous speed limit for next iteration
        self.previous_speed_limit = self.current_speed_limit
        
        return float(total_reward)

    def _calculate_speed_smoothness(self):
        """
        Calculate speed smoothness reward based on variance in speed history.
        Lower variance indicates better traffic harmonization.
        """
        if len(self.speed_history) < 2:
            return 0.0
        
        speed_variance = np.var(list(self.speed_history))
        # Normalize variance and invert (lower variance = higher reward)
        # Assuming max reasonable variance of 400 (20 m/s std deviation)
        max_variance = 400.0
        smoothness = max(0.0, 1.0 - (speed_variance / max_variance))
        return smoothness

    def _calculate_speed_trend(self):
        """
        Calculate speed trend over recent history.
        Positive trend indicates improving conditions, negative indicates deterioration.
        """
        if len(self.speed_history) < 3:
            return 0.0
        
        speeds = list(self.speed_history)
        # Simple linear regression slope calculation
        n = len(speeds)
        x = np.arange(n)
        
        # Calculate trend slope
        x_mean = np.mean(x)
        y_mean = np.mean(speeds)
        
        numerator = np.sum((x - x_mean) * (speeds - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return float(slope)

    def _preprocess_state(self, raw_state):
        """
        Normalize raw observation state vector to [0,1] range for DQN input.

        Args:
            raw_state (np.ndarray): Raw observation from environment step.

        Returns:
            np.ndarray: Normalized state vector as float32.
        """
        avg_speed = np.clip(raw_state[0], 0, MAX_SPEED_MPS) / MAX_SPEED_MPS
        flow_upstream = np.clip(raw_state[1], 0, MAX_FLOW) / MAX_FLOW
        flow_smoothed = np.clip(raw_state[2], 0, MAX_FLOW) / MAX_FLOW
        queue_length = np.clip(raw_state[3], 0, MAX_QUEUE_LENGTH_FOR_CRITICAL_SECTION) / MAX_QUEUE_LENGTH_FOR_CRITICAL_SECTION
        
        # Speed trend normalization: clip to [-1,1], then scale to [0,1]
        speed_trend = np.clip(raw_state[4], -SPEED_TREND_CLIP, SPEED_TREND_CLIP)
        speed_trend_norm = (speed_trend + SPEED_TREND_CLIP) / (2 * SPEED_TREND_CLIP)
        
        occupancy = np.clip(raw_state[5], 0, 1)  # already fraction
        speed_limit = np.clip(raw_state[6], 50, 130) / 130.0
        
        normalized_state = np.array([
            avg_speed,
            flow_upstream,
            flow_smoothed,
            queue_length,
            speed_trend_norm,
            occupancy,
            speed_limit
        ], dtype=np.float32)
        
        return normalized_state

    def _apply_vsl_enforcement(self, speed_limit_kmh):
        """
        Apply Variable Speed Limit enforcement based on configured mode.
        
        Args:
            speed_limit_kmh (float): Speed limit in km/h
        """
        speed_limit_ms = speed_limit_kmh / 3.6  # Convert to m/s
        
        if self.vsl_enforcement == "recommend":
            # Option 3: Only set maximum allowed speed for the lane
            for segId in seg_1_before:
                traci.lane.setMaxSpeed(segId, speed_limit_ms)
            logger.debug(f"VSL Mode {self.vsl_enforcement}: Set lane max speed to {speed_limit_kmh} km/h")
            
        elif self.vsl_enforcement == "all_vehicles":
            # Option 1: Force all vehicles to obey speed limit immediately
            for segId in seg_1_before:
                # Set lane max speed
                traci.lane.setMaxSpeed(segId, speed_limit_ms)
                
                # Force all vehicles in this lane to obey the new speed limit
                veh_ids = traci.lane.getLastStepVehicleIDs(segId)
                for veh_id in veh_ids:
                    try:
                        # Set vehicle speed to the new speed limit
                        traci.vehicle.setSpeed(veh_id, speed_limit_ms)
                    except Exception as e:
                        logger.debug(f"Could not set speed for vehicle {veh_id}: {e}")
            
            logger.debug(f"VSL Mode {self.vsl_enforcement}: Forced all vehicles to {speed_limit_kmh} km/h")
            
        elif self.vsl_enforcement == "electric_only":
            # Option 2: Force only electric_passenger vehicles to obey speed limit
            for segId in seg_1_before:
                # Set lane max speed
                traci.lane.setMaxSpeed(segId, speed_limit_ms)
                
                # Force only electric_passenger vehicles to obey the new speed limit
                veh_ids = traci.lane.getLastStepVehicleIDs(segId)
                for veh_id in veh_ids:
                    try:
                        # Check if vehicle type is electric_passenger
                        veh_type = traci.vehicle.getTypeID(veh_id)
                        if veh_type == "electric_passenger":
                            traci.vehicle.setSpeed(veh_id, speed_limit_ms)
                    except Exception as e:
                        logger.debug(f"Could not check/set speed for vehicle {veh_id}: {e}")
            
            logger.debug(f"VSL Mode {self.vsl_enforcement}: Forced electric_passenger vehicles to {speed_limit_kmh} km/h")
            
        else:
            logger.warning(f"Unknown VSL enforcement mode: {self.vsl_enforcement}. Using recommend.")
            # Fallback to recommend
            for segId in seg_1_before:
                traci.lane.setMaxSpeed(segId, speed_limit_ms)

    def _close_sumo(self, reason: str):
        """Safely closes the TraCI connection and terminates the SUMO process."""
        log_id = self._get_sumo_log_identifier()
        logger.debug(f"Closing SUMO for {log_id} due to: {reason}")
        
        if traci.isLoaded():
            try:
                traci.close(wait=False)
                logger.debug(f"TraCI connection closed for {log_id}.")
            except Exception as e:
                logger.warning(f"Exception during traci.close() for {log_id}: {e}")
        
        if self.sumo_process:
            # Check if the process is still running before trying to terminate
            if self.sumo_process.poll() is None: # poll() is None if process is running
                try:
                    logger.debug(f"Terminating SUMO process PID {self.sumo_process.pid} for {log_id}.")
                    self.sumo_process.terminate()
                    # +++ NEW: Wait for termination and capture final output +++
                    try:
                        # Wait for 5 seconds for the process to terminate gracefully
                        out, err = self.sumo_process.communicate(timeout=5)
                        if err:
                            logger.warning(f"Final SUMO stderr on close for {log_id}: {err.decode().strip()}")
                        if out:
                            logger.debug(f"Final SUMO stdout on close for {log_id}: {out.decode().strip()}")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"SUMO process PID {self.sumo_process.pid} did not terminate in time, killing.")
                        self.sumo_process.kill()
                        # Capture output after killing
                        out, err = self.sumo_process.communicate()
                        if err:
                            logger.error(f"Final SUMO stderr after kill for {log_id}: {err.decode().strip()}")

                    logger.debug(f"SUMO process PID {self.sumo_process.pid} has been handled.")

                except Exception as e:
                    logger.error(f"Exception during SUMO process termination for {log_id}: {e}")
            else:
                # If the process already finished, it's good practice to still communicate()
                # to clear the stdout/stderr buffers and prevent deadlocks.
                out, err = self.sumo_process.communicate()
                if err:
                    logger.debug(f"SUMO process for {log_id} had already terminated. Final stderr: {err.decode().strip()}")

            self.sumo_process = None
        self.is_sumo_initialized = False

    def close(self):
        """Closes the environment and its SUMO instance."""
        self._close_sumo(f"env.close() called for {self._get_sumo_log_identifier()}")
        if hasattr(self.logger, 'save_to_csv') and isinstance(self.logger, TrafficDataLogger): # If using TrafficDataLogger per env
             self.logger.save_to_csv(filename=f"traffic_log_{self._get_sumo_log_identifier()}.csv")

class TrafficDataLogger:
    """
    Comprehensive data logger for traffic simulation and RL training.
    Designed for SUMO-based VSL control experiments with SB3 integration.
    """
    
    def __init__(self, model_name: str, log_dir: str):
        """
        Initialize the traffic data logger.
        
        Args:
            model_name (str): Name of the model for identification.
            log_dir (str or Path): The base directory for logs.
        """
        self.model_name = model_name
        self.default_speed_limit = 130
        
        # *** FIX APPLIED HERE ***
        # self.output_dir is now correctly initialized as a Path object.
        self.output_dir = Path(log_dir) / "traffic_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Consistently use the Path object to define the summary log path.
        self.summary_log_path = self.output_dir / f"summary_log_{self.model_name}.csv"

        # Initialize or reset data containers
        self.data = []
        self.episode_rewards = []
        self.reset()

    def log_step_data(self, simulation_time, current_speed_limit, flow_upstream, 
                     flow_downstream, occupancy, queue_length, reward, action, avg_speed_before):
        """
        Log data for a single simulation step.
        """
        if current_speed_limit != self.last_speed_limit:
            self.speed_limit_changes += 1
            self.last_speed_limit = current_speed_limit
        
        action_map = {0: -10, 1: -5, 2: 0, 3: 5, 4: 10}
        speed_change = action_map.get(int(action), 0)

        step_data = {
            'timestamp': datetime.now().isoformat(),
            'simulation_time': simulation_time,
            'step': self.step_count,
            'episode': self.episode_count,
            'current_speed_limit': current_speed_limit,
            'speed_change': speed_change,
            'action': int(action),
            'flow_upstream': flow_upstream,
            'flow_downstream': flow_downstream,
            'occupancy': occupancy,
            'queue_length': queue_length,
            'reward': reward,
            'cumulative_reward': self.total_reward + reward,
            'avg_speed_before_mps': avg_speed_before,
        }
        
        self.data.append(step_data)
        self.step_count += 1
        self.total_reward += reward

    def save_to_csv(self, filename: str):
        """
        Save logged step-by-step data to a CSV file.

        Args:
            filename (str): The name of the file to save (e.g., 'log.csv').
        """
        if not self.data:
            logger.warning(f"No step data to save for {filename}")
            return

        # *** FIX IS EFFECTIVE HERE ***
        # The '/' operator now works because self.output_dir is a Path object.
        filepath = self.output_dir / filename

        try:
            df = pd.DataFrame(self.data)
            df.to_csv(filepath, index=False)
            logger.info(f"Traffic data log saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving step data to {filepath}: {e}")

    def get_summary_statistics(self) -> dict:
        """
        Calculates a comprehensive summary for a completed simulation run.
        """
        if not self.data:
            logger.warning(f"No data logged for {self.model_name}; cannot generate summary.")
            return {}

        df = pd.DataFrame(self.data)
        
        # --- Basic & Mobility Metrics ---
        total_steps = len(df)
        total_sim_time_s = df['simulation_time'].max()
        avg_flow_vph = df['flow_downstream'].mean()
        flow_stability_std_dev = df['flow_downstream'].std()
        
        # --- Congestion & Safety Metrics ---
        avg_queue_m = df['queue_length'].mean()
        max_queue_m = df['queue_length'].max()
        avg_speed_kph = df['avg_speed_before_mps'].mean() * 3.6
        speed_variance = df['avg_speed_before_mps'].var()

        # --- Control Effort Metrics ---
        control_actions = df[df['speed_change'] != 0].shape[0]
        control_frequency_pct = (control_actions / total_steps) * 100 if total_steps > 0 else 0

        summary_stats = {
            'total_steps': total_steps,
            'total_sim_time_s': total_sim_time_s,
            'avg_flow_vph': avg_flow_vph,
            'flow_stability_std_dev': flow_stability_std_dev,
            'avg_queue_m': avg_queue_m,
            'max_queue_m': max_queue_m,
            'avg_speed_kph': avg_speed_kph,
            'speed_variance': speed_variance,
            'total_control_actions': control_actions,
            'control_frequency_pct': control_frequency_pct,
        }
        
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in summary_stats.items()}

    def reset(self):
        """Resets the logger for a new episode or evaluation run."""
        self.start_time = time.time()
        self.data.clear()
        self.last_speed_limit = self.default_speed_limit
        self.step_count = 0
        self.total_reward = 0.0
        self.speed_limit_changes = 0
        self.best_reward = float('-inf')
        self.episode_count = 0
        logger.debug(f"TrafficDataLogger for model {self.model_name} has been reset.")

class TensorboardCallback(BaseCallback):
    def __init__(self, env, model, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.env = env  # Store the environment
        self.model = model  # Store the model

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            env = self.training_env.envs[0]  # assuming single env or first env
            if hasattr(env, "speed_history") and env.speed_history:
                mean_speed = env.speed_history[-1]
                self.logger.record("env/mean_speed", mean_speed)
        return True

class CustomMetricsCallback(BaseCallback):
    """
    A custom callback that logs key traffic metrics from the environment to TensorBoard.
    This version is compatible with SubprocVecEnv.
    """
    def __init__(self, verbose=0):
        super(CustomMetricsCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # 'locals' contains all local variables from the model's 'learn' method
        # 'infos' is a list of info dicts from each environment in the VecEnv
        infos = self.locals.get("infos", [])
        
        for i, info in enumerate(infos):
            # The 'final_info' key is added by the Monitor wrapper when an episode ends
            # This ensures we only log at the end of an episode, providing a stable summary
            if "final_info" in info:
                final_info = info["final_info"]
                
                # Log key performance indicators (KPIs) for the i-th environment
                # The 'custom' prefix helps group these in TensorBoard
                self.logger.record(f'custom/env_{i}/flow_downstream', final_info.get('flow_downstream', 0))
                self.logger.record(f'custom/env_{i}/avg_speed_before', final_info.get('avg_speed_before', 0))
                self.logger.record(f'custom/env_{i}/queue_length_upstream', final_info.get('queue_length_upstream', 0))
                self.logger.record(f'custom/env_{i}/collisions', final_info.get('collisions', 0))

                # Log the mean reward for the episode
                self.logger.record(f'custom/env_{i}/ep_reward', info.get('r', 0))

                # Example of a derived metric
                throughput_efficiency = final_info.get('flow_downstream', 0) / MAX_FLOW
                self.logger.record(f'custom/env_{i}/throughput_efficiency', throughput_efficiency)

        return True


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Main entry point for running the DRL VSL environment with SUMO. """

if __name__ == '__main__':
    # from https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        logger.info("SUMO environment is not set up correctly.")

    option = 2
    
    # Option 1: Run a single training with tuned parameters
    if option == 1:
        algo_to_use = "DQN"
        vsl_enforce_mode = "electric_only" 
        reward_used = "balanced"
        config_model_name = f"{algo_to_use}_{reward_used}_{vsl_enforce_mode}"

        optimal_params = ENHANCED_HYPERPARAMS["DQN"].copy()
        # optimal_params["gamma"] = 0.95 # Override gamma to 0.95 for this run FIXME: Temp debug, remove later

        create_sumocfg(config_model_name, vsl_enforce_mode)  # Add vsl_mode parameter
        train_model(algorithm=algo_to_use, 
                    reward_function=reward_used,
                    hyperparams=optimal_params,
                    vsl_enforcement=vsl_enforce_mode)

    # Option 2: Run parallel training for all combinations of reward functions and VSL enforcement modes
    elif option == 2:
        algo_to_use = "DQN"
        logger.info("Starting parallel training for all combinations using tuned or default parameters.")
        
        parallel_training_sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', SUMO_EXE_GUI)

        # Use the same lists as for tuning, or define them if option 3 wasn't run
        reward_functions = ["mobility", "safety", "balanced"] # ["mobility", "safety", "balanced"]
        vsl_enforcements = ["electric_only", "recommend"] # ["all_vehicles", "electric_only", "recommend"]

        use_hyperparams_by_optuna = False
        all_combinations_params_for_training = []
        process_counter = 0
        for r_fn_train in reward_functions:
            for vsl_m_train in vsl_enforcements:
                all_combinations_params_for_training.append((r_fn_train, vsl_m_train, process_counter, algo_to_use, parallel_training_sumo_binary, use_hyperparams_by_optuna))
                process_counter += 1
        
        num_parallel_training_processes = min(len(all_combinations_params_for_training), mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1) 
        logger.info(f"Running {len(all_combinations_params_for_training)} training combinations using up to {num_parallel_training_processes} parallel processes.")

        if sys.platform.startswith("win") or sys.platform.startswith("darwin"):
             mp.set_start_method('spawn', force=True)

        training_processes = []
        # Results list for training processes (if needed, currently run_training_for_combination logs its own success/failure)
        # training_results = mp.Manager().list() 
        
        active_training_processes = []
        for i, args_train in enumerate(all_combinations_params_for_training):
            p_train = mp.Process(target=run_training_for_combination, args=(args_train,))
            p_train.start()
            active_training_processes.append(p_train)
            
            if len(active_training_processes) >= num_parallel_training_processes:
                for proc_to_join in active_training_processes:
                    proc_to_join.join()
                active_training_processes = []

        for p_train in active_training_processes: # Join any remaining
            p_train.join()

        logger.info("Parallel training run finished for all combinations.")
        # for res_train in training_results: # If using a results list
        #     logger.info(res_train)
    
    # Option 3: Evaluate a trained model with tuned parameters
    elif option == 3:
        algo_to_use = "DQN"
        vsl_enforce_mode = "electric_only" 
        reward_used = "balanced"
        # Evaluate the trained model
        test_model(algorithm=algo_to_use, reward_function=reward_used, vsl_enforcement=vsl_enforce_mode)  # Add vsl_mode parameter

    """
    # Option 4: Quick test for TrafficDataLogger
    elif option == 4:
        print("--- Quick Test for TrafficDataLogger (Option 4) ---")
        test_log_dir = log_dir=Path("./logs/temp_test_logs") 
        test_identifier = "QuickTest_DQN"
        # test_id = 999
        try:
            logger_instance = TrafficDataLogger(
                model_name=test_identifier,
                log_dir=test_log_dir,
                # test_id  
            )
            
            print(f"  Logger instantiated.")
            
            summary_path_value = getattr(logger_instance, 'summary_log_path', 'NOT SET')
            print(f"  summary_log_path: {summary_path_value}")

            if summary_path_value not in ['NOT SET', None]:
                # Check if the directory for the summary log was created by the logger
                summary_dir = os.path.dirname(summary_path_value)
                if os.path.exists(summary_dir):
                    print(f"  Directory for summary_log_path exists: {summary_dir}")
                else:
                    print(f"  WARNING: Directory for summary_log_path does NOT exist: {summary_dir}")
            
        except NameError:
            print("  ERROR: TrafficDataLogger class not found. Ensure it's defined or imported.")
        except Exception as e:
            print(f"  ERROR during TrafficDataLogger quick test: {e}")
            # import traceback # Uncomment for full traceback if needed
            # traceback.print_exc()
        
        print("--- End of Quick Test ---")
    """

"""
Accepted limitations and Future Work:
- ✔️ [Works now] SUMO withough GUI is not supported in this environment, so GUI-based SUMO binary is used.
- 
"""