import logging
from stable_baselines3 import DQN
import torch.nn as nn
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from flow_gen import *
from gymnasium.wrappers import TimeLimit
import gymnasium as gym
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from datetime import datetime 
import psutil
from time import sleep
import traci
from traci import FatalTraCIError, TraCIException
import subprocess
import sys
from pathlib import Path
from collections import deque
import pandas as pd
import optuna
import glob
import time
import json
import multiprocessing as mp # Added for parallel processing
from itertools import product # Added for generating combinations

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the log level to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    handlers=[  # Handlers determine where logs are sent
        logging.StreamHandler()  # Output logs to stderr (default)
    ]
)

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
base_train_sumo_port = 8000
base_eval_sumo_port = 9000
""" Curriculum learning for the DQN agent, which means gradually increasing the difficulty of the training scenarios. """
interval_length_h = 2 # hours
num_of_intervals = 10
num_test_envs_per_model = 1
num_train_envs_per_model = 1
num_envs_per_model = num_train_envs_per_model + num_test_envs_per_model
interval_length = 60 * interval_length_h
sumoExecutable_gui = 'sumo-gui.exe' if os.name == 'nt' else 'sumo-gui'
sumoExecutable_nogui = 'sumo.exe' if os.name == 'nt' else 'sumo'
sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', sumoExecutable_gui) # Default to GUI

HYPER_PARAM_SIM_LENGTH = 1800 # Simulation length for hyperparameter tuning [s]
HYPER_PARAM_OPTUNA_STUD_TIMEOUT = 60 # seconds for hyperparameter tuning
HYPER_PARAM_MODEL_STEPS = 100
MAX_OCCUPANCY = 100.0  # Occupancy percentage
MAX_FLOW = 7200.0      # vehicles/hour (theoretical maximum for 2 lanes)
MAX_SPEED_DIFF = 80.0  # km/h (130 - 50)
MAX_QUEUE_LENGTH = 500 # vehicles (adjust based on your segment length)
OBSERVATION_SPACE_SIZE = 7
PROGRESS_BAR_ENABLED = False  # Enable progress bar for training

# Enhanced hyperparameters based on traffic control research
ENHANCED_HYPERPARAMS = {
    "DQN": {
        "learning_rate": 0.0001,
        "buffer_size": 100000,
        "batch_size": 32,
        "target_update_interval": 5000,
        "exploration_fraction": 0.15,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.02,
        "learning_starts": 10000,
        "train_freq": 4,
        "gradient_steps": 1,
        "tau": 1.0,
        "gamma": 0.995,
        "net_arch": [512, 256, 128]
    }
}

def create_sumocfg(model, vsl_enforcement="lane_only", model_idx_offset=0):
    sumocfg_template = """<?xml version="1.0" encoding="UTF-8"?>
    <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
        <input>
            <net-file value="3_2_merge.net.xml"/>
            <route-files value="generated_flows_{model}_{index}.rou.xml"/>
            <additional-files value="loops_detectors.add.xml"/>
            <gui-settings-file value="colored.view.xml"/>
        </input>
        <processing>
            <!-- Vehicles can occupy fractional positions within a lane, enabling smoother lateral movements.
                 This is important for zipper merging, as vehicles need to adjust their positions dynamically -->
            <lateral-resolution value="0.2"/>
        </processing>
    </configuration>
    """

    output_dir = "./traffic_environment/sumo"
    os.makedirs(output_dir, exist_ok=True)

    # Generate configuration files
    for i in range(num_envs_per_model):
        # Ensure index is unique even if model_name is the same for SubprocVecEnv instances
        actual_index = i + model_idx_offset
        filename = f"3_2_merge_{model}_{actual_index}.sumocfg"
        filepath = os.path.join(output_dir, filename)
        
        # Format the template with current model and index
        content = sumocfg_template.format(model=model, index=actual_index)
        
        # Write the content to the file
        with open(filepath, 'w') as file:
             file.write(content)
        
        logging.debug(f"Created {filepath}")

def train_env_constructor(idx, model_name, num_of_episodes, reward_fn, vsl_enforcement="lane_only", sumo_port_to_use=None, sumo_binary_to_use=None):
    def _init():
        # Use provided port or default from global, adjusted by idx
        port_for_env = sumo_port_to_use if sumo_port_to_use is not None else base_train_sumo_port + idx
        
        env = Monitor(TrafficEnv(port=port_for_env,
                                model_name=model_name,
                                model_idx=idx, # model_idx is specific to this sub-process env
                                op_mode="train",
                                base_gen_car_distrib=["uniform", 2000],
                                num_of_episodes=num_of_episodes,
                                reward_fn=reward_fn,
                                vsl_enforcement=vsl_enforcement,
                                sumo_binary_path_override=sumo_binary_to_use))
        return env
    return _init

def eval_env_constructor(model_name, reward_fn, vsl_enforcement="lane_only", sumo_port_to_use=None, sumo_binary_to_use=None):
    def _init():
        # Use provided port or default from global
        port_for_eval_env = sumo_port_to_use if sumo_port_to_use is not None else base_eval_sumo_port
        # model_idx for eval env can be fixed, e.g., num_envs_per_model -1, or a dedicated high number
        eval_model_idx = num_envs_per_model -1 # Or a distinct ID like 999

        env = Monitor(TimeLimit(TrafficEnv(port=port_for_eval_env,
                                            model_name=model_name, # model_name is unique per parallel run
                                            model_idx=eval_model_idx, 
                                            op_mode="eval",
                                            base_gen_car_distrib=["uniform", 3000],
                                            num_of_episodes=1,
                                            reward_fn=reward_fn,
                                            vsl_enforcement=vsl_enforcement,
                                            sumo_binary_path_override=sumo_binary_to_use),
                                max_episode_steps=interval_length))
        return env
    return _init

def train_model(algorithm,
                reward_function="balanced", 
                num_of_episodes=7,
                use_enhanced_params=True,
                custom_params=None,
                vsl_enforcement="lane_only",
                process_train_base_port=None,
                process_eval_base_port=None,
                sumo_binary_to_use=None):
    """Fixed version with correct SB3 2.6.0 parameter names. DQN only."""
    # Only DQN supported
    params = ENHANCED_HYPERPARAMS["DQN"].copy() if use_enhanced_params else {
        "learning_rate": 1e-4,
        "buffer_size": 100000,
        "batch_size": 128,
        "target_update_interval": 1000,
        "exploration_fraction": 0.1,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.01,
        "learning_starts": 1000,
        "train_freq": 4,
        "gradient_steps": 1,
        "tau": 1.0,
        "gamma": 0.99,
        "net_arch": [256, 256, 128]
    }
    if custom_params:
        params.update(custom_params)
        logging.info(f"Applied custom parameter overrides: {custom_params}")

    steps_per_episode = 504000 // 60  
    total_timesteps = steps_per_episode * num_of_episodes
    eval_timesteps = steps_per_episode // 4

    model_name = f"{algorithm}_{reward_function}_{vsl_enforcement}"
    log_dir = f"./logs/{model_name}/"
    model_dir = f"./rl_models/{model_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Determine base ports for SubprocVecEnv if provided
    # If num_train_envs_per_model > 1, each sub-env needs its own port.
    # The process_train_base_port is the starting port for this train_model call.
    current_train_base_port = process_train_base_port if process_train_base_port is not None else base_train_sumo_port
    current_eval_base_port = process_eval_base_port if process_eval_base_port is not None else base_eval_sumo_port

    train_env = SubprocVecEnv([
        train_env_constructor(i, model_name, num_of_episodes, reward_function, vsl_enforcement,
                              sumo_port_to_use=current_train_base_port + i, # Each sub-env gets a unique port
                              sumo_binary_to_use=sumo_binary_to_use)
        for i in range(num_train_envs_per_model)
    ])
    env_eval = SubprocVecEnv([eval_env_constructor(model_name, reward_function, vsl_enforcement,
                                                  sumo_port_to_use=current_eval_base_port, # Eval env gets its own port
                                                  sumo_binary_to_use=sumo_binary_to_use)])

    policy_kwargs = dict(
        net_arch=params.pop("net_arch", [256, 256, 128]), # Deeper network for complex traffic patterns
        activation_fn=nn.ReLU
    )

    model = DQN("MlpPolicy", train_env, 
               learning_rate=params["learning_rate"],
               buffer_size=params["buffer_size"],
               batch_size=params["batch_size"],
               target_update_interval=params["target_update_interval"],
               exploration_fraction=params["exploration_fraction"],
               exploration_initial_eps=params["exploration_initial_eps"],
               exploration_final_eps=params["exploration_final_eps"],
               learning_starts=params["learning_starts"],
               train_freq=params["train_freq"],
               gradient_steps=params["gradient_steps"],
               tau=params["tau"],
               gamma=params["gamma"],
               policy_kwargs=policy_kwargs,
               verbose=1, tensorboard_log=log_dir, device='cuda')

    logging.info(f"Training DQN with parameters: {params}")

    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    checkpoint_cb = CheckpointCallback(
        save_freq=eval_timesteps,
        save_path=model_dir,
        name_prefix=f"rl_model_{model_name}",
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=1
    )

    no_improve_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=1,
        min_evals=3,
        verbose=1
    )

    eval_cb = EvalCallback(
        env_eval,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=eval_timesteps,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        callback_after_eval=no_improve_cb,
        verbose=1
    )

    # Training loop
    try:
        model.learn(total_timesteps=total_timesteps,
                    callback=[checkpoint_cb, eval_cb],
                    progress_bar=PROGRESS_BAR_ENABLED,
                    reset_num_timesteps=False)
        model.save(os.path.abspath(f"./rl_models/{model_name}/{model_name}.zip"))
    except KeyboardInterrupt:
        logging.warning(f"Training for {model_name} interrupted by user.")
    except Exception as e:
        logging.error(f"Error during training for {model_name}: {e}")
    finally:
        train_env.close()
        env_eval.close()
        logging.info(f"Finished training for {model_name}")

def test_model(algorithm, reward_function, vsl_enforcement="lane_only"):
    """Test a trained DQN model with comprehensive evaluation."""
    model_name = f"{algorithm}_{reward_function}_{vsl_enforcement}"
    try:
        model_path = f"rl_models/{model_name}/best_model"
        model = DQN.load(model_path)
    except FileNotFoundError:
        logging.warning(f"Best model not found, loading checkpoint...")
        checkpoint_path = f"rl_models/{model_name}/rl_model_{model_name}_final.zip"
        model = DQN.load(checkpoint_path)

    env = TrafficEnv(port=base_eval_sumo_port,
                     model_name=model_name,
                     model_idx=0,
                     op_mode="test",
                     base_gen_car_distrib=["bimodal", 3],
                     reward_fn=reward_function,
                     vsl_enforcement=vsl_enforcement)  # Add VSL enforcement parameter

    obs, _ = env.reset()
    total_reward = 0
    step_count = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        if done or truncated:
            break

    env.close()

    print(f"Test Results for {model_name}:")
    print(f"Total Steps: {step_count}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Average Reward: {total_reward/step_count:.3f}")
    print(f"Final Flow Rate: {info.get('flow_downstream', 0):.1f} veh/h")

def tune_hyperparameters(algorithm, reward_function, n_trials=20, warm_start_file=None, vsl_enforcement="lane_only",
                         tuning_process_base_port=None,
                         sumo_binary_to_use=None):
    """
    Efficient hyperparameter tuning using existing infrastructure. DQN only.
    """
    # Generate unique warm start file name if not provided
    if warm_start_file is None:
        os.makedirs("rl_models/optuna_params", exist_ok=True)
        warm_start_file = os.path.join("rl_models", "optuna_params", f"best_optuna_params_{algorithm}_{reward_function}_{vsl_enforcement}.json")
    
    scenario_configs = [
        # {"id": 100, "demand": 2000, "pattern": "uniform"},
        # {"id": 101, "demand": 2500, "pattern": "uniform"}, 
        {"id": 102, "demand": 3000, "pattern": "uniform"},
        {"id": 103, "demand": 3500, "pattern": "uniform"}
    ]
    model_name = f"{algorithm}_tune_{reward_function}_{vsl_enforcement}"
    output_dir_sumo = Path("./traffic_environment/sumo")
    output_dir_sumo.mkdir(parents=True, exist_ok=True)

    sumocfg_template = """<?xml version="1.0" encoding="UTF-8"?>
    <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
        <input>
            <net-file value="3_2_merge.net.xml"/>
            <route-files value="generated_flows_{model}_{index}.rou.xml"/>
            <additional-files value="loops_detectors.add.xml"/>
            <gui-settings-file value="colored.view.xml"/>
        </input>
        <processing>
            <lateral-resolution value="0.2"/>
        </processing>
    </configuration>
    """

    for config in scenario_configs:
        flow_generation_fix_num_veh(
            model_name, 
            config["id"],
            config["demand"], 
            num_of_hrs=1,
            num_of_episodes=1, 
            num_of_intervals=1, 
            op_mode="train"
        )
        cfg_filename = f"3_2_merge_{model_name}_{config['id']}.sumocfg"
        cfg_filepath = output_dir_sumo / cfg_filename
        cfg_content = sumocfg_template.format(model=model_name, index=config['id'])
        with open(cfg_filepath, 'w') as file:
            file.write(cfg_content)
        logging.debug(f"Created {cfg_filepath} for tuning scenario id {config['id']}")

    def objective(trial):
        net_arch_str_suggestion = trial.suggest_categorical("net_arch_str", ["256,256", "512,256", "256,128,64"])
        net_arch_list = [int(x) for x in net_arch_str_suggestion.split(',')]
        policy_kwargs = dict(net_arch=net_arch_list, activation_fn=nn.ReLU)
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "buffer_size": trial.suggest_categorical("buffer_size", [50000, 100000, 200000]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "target_update_interval": trial.suggest_int("target_update_interval", 1000, 10000),
            "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.3),
            "exploration_initial_eps": trial.suggest_float("exploration_initial_eps", 0.5, 1.0),
            "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1),
            "learning_starts": trial.suggest_categorical("learning_starts", [1000, 5000]),
            "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]),
            "gradient_steps": trial.suggest_categorical("gradient_steps", [1, -1]),
            "tau": trial.suggest_float("tau", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.95, 0.999),
        }
        total_reward = 0
        current_tuning_base_port = tuning_process_base_port if tuning_process_base_port is not None else base_train_sumo_port

        for config_item in scenario_configs:
            env = None  # Initialize env to None
            try:
                port_for_tuning_env = current_tuning_base_port + trial.number * len(scenario_configs) + config_item["id"] % len(scenario_configs)
                env = TrafficEnvForTuning(
                    port=port_for_tuning_env, model_name=model_name, model_idx=config_item["id"],
                    op_mode="train", base_gen_car_distrib=["uniform", config_item["demand"]],
                    num_of_episodes=1, reward_fn=reward_function, skip_flow_generation=True,
                    vsl_enforcement=vsl_enforcement, sumo_binary_path_override=sumo_binary_to_use
                )
                model = DQN("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs, **params)
                model.learn(total_timesteps=HYPER_PARAM_MODEL_STEPS, progress_bar=False)
                obs, _ = env.reset() # Reset for evaluation if needed by your logic
                episode_reward = 0
                for step_eval in range(HYPER_PARAM_MODEL_STEPS): # Or a different number of eval steps
                    action_eval, _ = model.predict(obs, deterministic=True)
                    obs, reward_eval, done_eval, truncated_eval, _ = env.step(action_eval)
                    episode_reward += reward_eval
                    if done_eval or truncated_eval:
                        break
                total_reward += episode_reward
            except Exception as e:
                logging.error(f"Trial {trial.number} scenario {config_item['id']} for {model_name} failed: {e}")
                return float('-inf')
            finally:
                if env is not None: # Check if env was successfully created
                    try:
                        logging.debug(f"Closing env for trial {trial.number}, scenario {config_item['id']} in finally block.")
                        env.close() # This calls _robust_close_sumo
                    except Exception as close_e:
                        logging.error(f"Error during env.close() in finally for trial {trial.number}, scenario {config_item['id']}: {close_e}", exc_info=True)
                    # del env # Optional: Python's GC should handle it, but can be explicit
        if not scenario_configs: # Avoid division by zero if scenario_configs is empty
            return 0.0
        return total_reward / len(scenario_configs)

    study = optuna.create_study(direction='maximize')
    try:
        with open(warm_start_file, "r") as f:
            prev_best_params = json.load(f)
        study.enqueue_trial(prev_best_params)
        logging.info(f"Enqueued previous best parameters for warm start: {prev_best_params}")
    except Exception as e:
        logging.info(f"No previous Optuna params for warm start: {e}")

    study.optimize(objective, n_trials=n_trials, timeout=HYPER_PARAM_OPTUNA_STUD_TIMEOUT)

    # Path for the single, shared Optuna params file (used for warm start if it exists)
    shared_optuna_params_filename = "best_optuna_params.json"
    shared_optuna_params_path = os.path.join("rl_models", "optuna_params", "best_optuna_params.json")
    
    # Warm start from the SHARED file if it exists
    if os.path.exists(shared_optuna_params_path):
        try:
            with open(shared_optuna_params_path, "r") as f:
                prev_best_params = json.load(f)
            study.enqueue_trial(prev_best_params)
            logging.info(f"Enqueued previous best parameters from SHARED {shared_optuna_params_path} for warm start.")
        except Exception as e:
            logging.info(f"No previous Optuna params for warm start from SHARED {shared_optuna_params_path}: {e}")
    else:
        logging.info(f"SHARED Optuna params file {shared_optuna_params_path} not found. Starting fresh study.")


    study.optimize(objective, n_trials=n_trials, timeout=HYPER_PARAM_OPTUNA_STUD_TIMEOUT)

    logging.info(f"Optuna study for {model_name} completed. Best params: {study.best_params}")

    # Save to the SHARED generic file path
    os.makedirs(os.path.dirname(shared_optuna_params_path), exist_ok=True)
    with open(shared_optuna_params_path, "w") as f:
        json.dump(study.best_params, f)
    logging.info(f"Saved best Optuna params to SHARED location: {shared_optuna_params_path}")

    # Optional: Save to a specific file as well if needed for other purposes or detailed tracking
    if warm_start_file: # This was the originally passed or generated specific filename
        specific_params_dir = os.path.dirname(warm_start_file)
        if specific_params_dir and not os.path.exists(specific_params_dir):
            os.makedirs(specific_params_dir, exist_ok=True)
        try:
            with open(warm_start_file, "w") as f:
                json.dump(study.best_params, f)
            logging.info(f"Saved best Optuna params also to specific file: {warm_start_file}")
        except Exception as e_specific_save:
            logging.error(f"Could not save to specific warm_start_file {warm_start_file}: {e_specific_save}")


    # --- CRITICAL DELAY BEFORE FILE CLEANUP ---
    delay_before_cleanup = 10 # seconds
    logging.info(f"Waiting {delay_before_cleanup} seconds before cleaning up tuning files for {model_name} to ensure SUMO processes are closed...")
    time.sleep(delay_before_cleanup)
    # --- END CRITICAL DELAY ---

    patterns_to_clean = [
        f"generated_flows_{model_name}_*.rou.xml",
        f"3_2_merge_{model_name}_*.sumocfg"
    ]

    def wait_for_file_release(filepath_to_clean, timeout=10): # Increased default timeout
        start_time_fr = time.time()
        file_path_obj_fr = Path(filepath_to_clean)

        if not file_path_obj_fr.exists():
            logging.debug(f"File {filepath_to_clean} does not exist. No need to remove.")
            return True

        logging.debug(f"Attempting to remove {filepath_to_clean}...")
        while time.time() - start_time_fr < timeout:
            try:
                os.remove(filepath_to_clean)
                logging.debug(f"Successfully removed: {filepath_to_clean}")
                return True
            except FileNotFoundError: # If removed by another process or in a previous attempt
                logging.debug(f"File {filepath_to_clean} already gone (FileNotFoundError during retry).")
                return True
            except PermissionError as e_perm_fr: # Specifically catch PermissionError (WinError 32)
                logging.warning(f"Could not remove {filepath_to_clean} due to PermissionError (likely in use): {e_perm_fr}. Retrying in 1s...")
                time.sleep(1)
            except Exception as e_fr: # Catch other potential OS errors
                logging.warning(f"Could not remove {filepath_to_clean} due to OS error: {e_fr}. Retrying in 1s...")
                time.sleep(1)
        
        logging.error(f"Failed to remove {filepath_to_clean} after {timeout} seconds. It might still be in use.")
        if file_path_obj_fr.exists(): # Check one last time
            logging.error(f"File {filepath_to_clean} STILL EXISTS. Listing active SUMO processes:")
            try:
                for proc in psutil.process_iter(['pid', 'name']): # Removed 'username' for brevity/permission
                    if 'sumo' in proc.info['name'].lower():
                        logging.error(f"  Potential SUMO culprit: PID {proc.info['pid']}, Name {proc.info['name']}")
            except (psutil.Error) as e_psutil: # Catch all psutil errors
                 logging.error(f"Could not list processes due to psutil error: {e_psutil}")
        return False

    for pattern in patterns_to_clean:
        for filepath_to_clean_glob in glob.glob(os.path.join("./traffic_environment/sumo/", pattern)):
            # Ensure we only clean files related to the current tuning scenarios
            if any(f"_{config['id']}." in filepath_to_clean_glob or f"_{config['id']}.sumocfg" in filepath_to_clean_glob for config in scenario_configs):
                 logging.debug(f"Targeting specific tuning file for cleanup: {filepath_to_clean_glob}")
                 wait_for_file_release(filepath_to_clean_glob)
    
    return study.best_params

def get_optimal_params(algorithm, traffic_density, episode_length):
    """
    Select optimal parameters based on traffic conditions and training requirements. DQN only.
    """
    base_params = ENHANCED_HYPERPARAMS["DQN"].copy()
    if traffic_density == "high":
        base_params["exploration_fraction"] = 0.2
        base_params["exploration_final_eps"] = 0.05
        base_params["learning_rate"] *= 0.5
    elif traffic_density == "low":
        base_params["learning_rate"] *= 1.5
        base_params["exploration_fraction"] = 0.1
    if episode_length == "long":
        base_params["gamma"] = 0.999
    elif episode_length == "short":
        base_params["gamma"] = 0.95
    return base_params

INITIAL_PARALLEL_PORT_BASE = 9500
PORTS_PER_PARALLEL_PROCESS = 50

def run_training_for_combination(config_tuple):
    reward_fn, vsl_mode, process_id, algo_used, parallel_sumo_binary = config_tuple
    
    base_port_for_this_process = INITIAL_PARALLEL_PORT_BASE + process_id * PORTS_PER_PARALLEL_PROCESS
    main_train_base_port = base_port_for_this_process + 150 
    main_eval_base_port = base_port_for_this_process + 150 + num_train_envs_per_model + 5

    config_model_name = f"{algo_used}_{reward_fn}_{vsl_mode}"
    
    logging.info(f"Process {process_id}: Starting combination {config_model_name}. Train Port Base: {main_train_base_port}, Eval Port Base: {main_eval_base_port}")

    best_params = None
    shared_optuna_params_filename = "best_optuna_params.json"
    shared_optuna_params_path = os.path.join("traffic_environment", shared_optuna_params_filename)

    if os.path.exists(shared_optuna_params_path):
        try:
            with open(shared_optuna_params_path, "r") as f:
                best_params = json.load(f)
            logging.info(f"Process {process_id}: Loaded shared best Optuna params from {shared_optuna_params_path} for {config_model_name}")
        except Exception as e:
            logging.warning(f"Process {process_id}: Could not load shared Optuna params from {shared_optuna_params_path}: {e}. Using default ENHANCED_HYPERPARAMS.")
            best_params = None # Fallthrough to default
    
    if not best_params:
        logging.info(f"Process {process_id}: Shared Optuna params file not found or failed to load from {shared_optuna_params_path}. Using default ENHANCED_HYPERPARAMS for {config_model_name}.")
        best_params = ENHANCED_HYPERPARAMS["DQN"].copy()
        # Ensure net_arch is a list if it was somehow a string in defaults (should not be)
        if "net_arch_str" in best_params:
             del best_params["net_arch_str"] # ENHANCED_HYPERPARAMS should have 'net_arch' as list

    if best_params and "net_arch_str" in best_params: # If loaded from JSON where it was saved as str
        net_arch_list = [int(x) for x in best_params["net_arch_str"].split(",")]
        best_params["net_arch"] = net_arch_list
        del best_params["net_arch_str"]
    elif not best_params:
        logging.error(f"Process {process_id}: best_params is None for {config_model_name} even after fallback. Critical error.")
        return f"Failure: {config_model_name} - best_params is None"

    try:
        # Create SUMO config
        logging.info(f"Process {process_id}: Creating SUMO config for {config_model_name}...")
        create_sumocfg(config_model_name, vsl_mode) 
                                           
        # Train Model
        logging.info(f"Process {process_id}: Training model {config_model_name} with params: {best_params}")
        train_model(algorithm=algo_used,
                    reward_function=reward_fn,
                    use_enhanced_params=False, # We are using loaded/default params
                    custom_params=best_params,
                    vsl_enforcement=vsl_mode,
                    process_train_base_port=main_train_base_port,
                    process_eval_base_port=main_eval_base_port,
                    sumo_binary_to_use=parallel_sumo_binary)
        
        logging.info(f"Process {process_id}: Successfully completed {config_model_name}")
        return f"Success: {config_model_name}"
    except Exception as e:
        logging.error(f"Process {process_id}: FAILED for {config_model_name}. Error: {e}", exc_info=True)
        return f"Failure: {config_model_name} - {e}"

def _robust_close_sumo(self, reason):
    # Identify which environment is closing for logging
    env_id_str = f"({self.effective_model_name_for_files}_{self.effective_model_idx_for_files})" \
                 if hasattr(self, 'effective_model_name_for_files') \
                 else f"(Tuning {self.model_name}_{self.model_idx})"

    logging.debug(f"Closing SUMO {env_id_str}: {reason}")
    
    # Attempt to save logs before closing SUMO
    try:
        if isinstance(self, TrafficEnvForTuning): # Specific logging for tuning
            if hasattr(self, 'logger') and len(self.logger.data) > 10: # Only save if significant data
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"tuning_data_{self.model_name}_{self.model_idx}_{timestamp}.csv"
                self.logger.save_to_csv(filename, include_summary=False)
        elif hasattr(self, 'logger'): # General TrafficEnv logging
             self.logger.save_to_csv(
                f"rl_train_{self.effective_model_name_for_files}_{self.effective_model_idx_for_files}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
            )
    except Exception as e:
        logging.warning(f"Could not save logs for {env_id_str} before SUMO close: {e}")

    if traci.isLoaded():
        try:
            traci.close(wait=False)
            logging.debug(f"traci.close() called for {env_id_str}.")
        except (FatalTraCIError, TraCIException, ConnectionResetError, BrokenPipeError, ImportError) as e:
            logging.debug(f"Exception during traci.close() for {env_id_str} (SUMO likely already gone or traci not available): {e}")

    self.is_sumo_initialized = False

    if self.sumo_process is not None:
        pid = self.sumo_process.pid
        logging.debug(f"SUMO process PID {pid} exists for {env_id_str}. Attempting to terminate.")
        try:
            if psutil.pid_exists(pid):
                parent = psutil.Process(pid)
                # Terminate children first (SUMO sometimes spawns sub-processes)
                for child in parent.children(recursive=True):
                    try:
                        logging.debug(f"Terminating child process {child.pid} of SUMO {pid} for {env_id_str}")
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass # Child already gone
                    except Exception as e_child_term:
                        logging.warning(f"Could not terminate child {child.pid} of SUMO {pid} for {env_id_str}: {e_child_term}")
                
                # Wait for children to terminate
                gone, alive = psutil.wait_procs(parent.children(recursive=True), timeout=2)
                for p_alive in alive:
                    logging.warning(f"Child process {p_alive.pid} of SUMO {pid} for {env_id_str} did not terminate, killing.")
                    try: p_alive.kill()
                    except psutil.NoSuchProcess: pass

                # Terminate parent SUMO process
                logging.debug(f"Terminating main SUMO process {pid} for {env_id_str}")
                parent.terminate()
                try:
                    parent.wait(timeout=5)
                    logging.debug(f"SUMO process {pid} terminated gracefully for {env_id_str}.")
                except psutil.TimeoutExpired:
                    logging.warning(f"SUMO process {pid} did not terminate in 5s for {env_id_str}. Forcing kill.")
                    if psutil.pid_exists(pid): # Check again before kill
                        parent.kill()
                        parent.wait(timeout=2)
                        logging.debug(f"SUMO process {pid} killed for {env_id_str}.")
                except Exception as e_wait:
                    logging.error(f"Error during SUMO process wait for {pid} ({env_id_str}): {e_wait}")
            else:
                logging.debug(f"SUMO process PID {pid} no longer exists for {env_id_str} (checked by psutil).")
        except psutil.NoSuchProcess:
            logging.debug(f"SUMO process PID {pid if pid else 'unknown'} (NoSuchProcess) already gone before explicit termination for {env_id_str}.")
        except Exception as e:
            logging.error(f"General error terminating SUMO process {pid if pid else 'None'} for {env_id_str}: {e}")
        finally:
            self.sumo_process = None
    else:
        logging.debug(f"No SUMO process to terminate for {env_id_str}.")
        
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Classes """
MAX_SPEED_MPS = 130 / 3.6       # 36.11 m/s approx
MAX_FLOW = 7200.0               # vehicles per hour
MAX_QUEUE_LENGTH = 500.0        # meters
SPEED_TREND_CLIP = 1.0          # max absolute slope value for clipping

class TrafficEnv(gym.Env):
    def __init__(self, port, model_name, model_idx, op_mode, base_gen_car_distrib, 
                 num_of_episodes=0, reward_fn="balanced", vsl_enforcement="lane_only",
                 sumo_binary_path_override=None):
        super(TrafficEnv, self).__init__()
        self.default_speed_limit = 130
        self.port = port
        self.model_name = model_name
        self.model_idx = model_idx
        self.effective_model_name_for_files = model_name
        self.effective_model_idx_for_files = model_idx
        self.sumo_binary_path_override = sumo_binary_path_override
        self.skip_flow_generation = False
        self.aggregation_time = 60  # [s] Data aggregation duration
        self.sumo_process = None
        self.sumo_max_retries = 3
        self.operation_mode = op_mode
        self.is_sumo_initialized = False  # Track SUMO initialization state
        self.collisions = []
        self.collisions_penalty = 0
        self.gen_car_distrib = base_gen_car_distrib
        self.logger = TrafficDataLogger(self.default_speed_limit)
        self.num_of_episodes = num_of_episodes
        self.reward_fn = reward_fn
        
        # Historical data for analysis
        self.flow_downstream_history = deque(maxlen=5)
        self.occupancy_downstream_history = deque(maxlen=5)
        self.speed_history = deque(maxlen=15)
        self.flow_smoothed = 0.0
        self.occupancy_smoothed = 0.0
        
        # Action and observation spaces
        self.action_space = gym.spaces.Discrete(3)
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
        
        # Set simulation length based on operation mode
        if self.operation_mode == "train":
            # For training: run for generated flow duration (504000 seconds)
            self.sim_length = 504000  # Full simulation time in seconds
        elif self.operation_mode == "eval":
            self.sim_length = int(interval_length * num_of_intervals)
        elif self.operation_mode == "test":
            self.sim_length = int(24 * 3600)  # 24 hours in seconds

        # Track a moving average of recent rewards. If this average falls below a threshold for a certain number of steps, terminate the episode early.
        self.reward_window = deque(maxlen=50)  # Track last 50 rewards
        self.reward_threshold = -5 # Threshold for early termination in tuning

        # VSL Enforcement Mode Configuration
        # Options: "all_vehicles", "electric_only", "lane_only"
        self.vsl_enforcement = vsl_enforcement
        
        # Aattributes for start_sumo customization
        self._sumo_start_context_prefix = ""  # e.g., "Tuning " for the child class
        self._default_sumo_binary_for_env = sumoBinary # Default for TrafficEnv
        self._sumo_retry_sleep_func = lambda attempt, max_retries: max_retries + attempt # Default retry logic

    def _get_sumo_log_identifier(self):
        """Helper to get a consistent identifier for SUMO instance logging."""
        # For TrafficEnvForTuning, effective_model_name_for_files includes "_tune_"
        # and effective_model_idx_for_files is the scenario_id.
        # For TrafficEnv, these are the main model name and sub-env index.
        return f"{self._sumo_start_context_prefix}{self.effective_model_name_for_files}_{self.effective_model_idx_for_files}"
    
    def start_sumo(self):
        """Initialize SUMO simulation - only start if not already running properly."""
        log_id = self._get_sumo_log_identifier()

        if self.is_sumo_initialized and self.sumo_process and psutil.pid_exists(self.sumo_process.pid):
            try:
                traci.simulation.getTime()
                logging.debug(f"SUMO ({log_id}) is already running and responsive.")
                return
            except (FatalTraCIError, TraCIException, ConnectionResetError, BrokenPipeError):
                logging.warning(f"SUMO process ({log_id}) exists but not responsive, restarting...")
                self.is_sumo_initialized = False # Mark for restart
        
        if self.sumo_process and psutil.pid_exists(self.sumo_process.pid):
            self.close_sumo(f"Restarting SUMO for initialization ({log_id})")
            sleep(3) # Give a bit more time for resources to free up
        elif self.sumo_process and not psutil.pid_exists(self.sumo_process.pid):
            logging.debug(f"SUMO process handle existed for {log_id} but PID was not found. Clearing handle.")
            self.sumo_process = None # Clear stale handle

        if traci.isLoaded():
            try:
                traci.close(wait=False)
                logging.debug(f"Closed existing TraCI connection before starting new SUMO instance for {log_id}.")
            except Exception as e:
                logging.warning(f"Error closing previous TraCI connection for {log_id}: {e}")
        
        for attempt in range(self.sumo_max_retries):
            try:
                port = self.port
                
                if not self.skip_flow_generation:
                    if self.gen_car_distrib[0] == 'uniform':
                        flow_generation_fix_num_veh(self.effective_model_name_for_files, self.effective_model_idx_for_files,
                                                    self.gen_car_distrib[1],
                                                    int(interval_length // 60),
                                                    self.num_of_episodes,
                                                    num_of_intervals,
                                                    self.operation_mode)
                    elif self.gen_car_distrib[0] == 'bimodal':
                        flow_generation(self.effective_model_name_for_files, self.effective_model_idx_for_files,
                                        bimodal_distribution_24h(self.gen_car_distrib[1]), 1)
                
                current_sumo_binary = self.sumo_binary_path_override if self.sumo_binary_path_override else self._default_sumo_binary_for_env
                
                sumo_cmd = [
                    current_sumo_binary, "-c",
                    f"./traffic_environment/sumo/3_2_merge_{self.effective_model_name_for_files}_{self.effective_model_idx_for_files}.sumocfg",
                    '--start',
                    "--default.emergencydecel=7",
                    '--random-depart-offset=3600',
                    "--remote-port", str(port),
                    "--step-length=0.1",
                    "--default.action-step-length=0.2",
                    f"--end={self.sim_length}",
                    "--quit-on-end",
                    "--no-step-log", 
                    "--no-warnings"  
                ]

                self.sumo_process = subprocess.Popen(sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                logging.info(f"Attempting to connect to SUMO ({log_id}) on port {port}")
                time.sleep(0.5) 
                traci.init(port=port, numRetries=5, host='127.0.0.1')
                logging.info(f"Successfully connected to SUMO ({log_id}) on port {port} for {self.sim_length}s simulation")
                self.is_sumo_initialized = True
                break
                
            except (FatalTraCIError, TraCIException, ConnectionRefusedError) as e:
                logging.error(f"Attempt {attempt + 1} to start/connect SUMO ({log_id}) failed: {e}")
                self.close_sumo(f"Failed to start/connect SUMO ({log_id}) on attempt {attempt+1}")
                if attempt < self.sumo_max_retries - 1:
                    sleep(self._sumo_retry_sleep_func(attempt, self.sumo_max_retries)) # Use customized retry sleep
                else:
                    logging.error(f"Max retries reached for starting SUMO ({log_id}). Raising exception.")
                    raise e

    def step(self, action):
        """Execute one step in the environment."""
        # Initialize SUMO if not already done
        if not self.is_sumo_initialized:
            self.start_sumo()
        
        # Check SUMO responsiveness
        try:
            current_time = traci.simulation.getTime()
        except (FatalTraCIError, TraCIException):
            logging.error("Lost connection to SUMO, restarting...")
            self.is_sumo_initialized = False
            self.start_sumo()
            current_time = traci.simulation.getTime()
        
        # Apply action: gradual speed limit changes
        speed_changes = [-5, 0, +5]
        previous_speed_limit = self.current_speed_limit
        self.current_speed_limit += speed_changes[action]
        
        # Invalid action penalty and clamping
        invalid_action_penalty = 0
        if (previous_speed_limit <= 50 and action == 0) or (previous_speed_limit >= 130 and action == 2):
            invalid_action_penalty = -1
        
        self.current_speed_limit = max(50, min(130, self.current_speed_limit))
        
        # Apply VSL enforcement using the new method
        self.apply_vsl_enforcement(self.current_speed_limit)
        
        # Initialize data collection variables
        flow_upstream_temp = 0
        flow_downstream_temp = 0
        queue_length_temp = 0
        mean_speeds_downstream = 0
        mean_speeds_upstream = 0
        occupancy_upstream_temp = 0
        
        # Simulation steps and data aggregation
        for step in range(self.aggregation_time):
            try:
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                self.simulation_step += 1
            except (FatalTraCIError, TraCIException):
                logging.error("Lost connection during simulation steps")
                # Reset environment instead of crashing
                return self.reset()
            
            # Collect traffic measurements
            flow_upstream_temp += traci.edge.getLastStepVehicleNumber("seg_0_before")
            flow_downstream_temp += traci.edge.getLastStepVehicleNumber("seg_0_after")
            
            # Queue length based on halting vehicles
            queue_length_temp += sum([
                traci.lane.getLastStepHaltingNumber(lane) * 7.5
                for lane in seg_1_before
            ])
            
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
        observation = self.preprocess_state(raw_observation)
                
        # Check termination conditions
        # End when simulation time reaches limit OR no more vehicles expected
        done = (current_time >= self.sim_length) or (traci.simulation.getMinExpectedNumber() <= 0) or (len(self.reward_window) == self.reward_window.maxlen and np.mean(self.reward_window) < self.reward_threshold)
        
        # Log data
        self.logger.log_step_data(
            current_time, self.current_speed_limit, self.flow_upstream,
            self.flow_downstream, self.occupancy_upstream, self.queue_length_upstream,
            reward, action
        )
        
        info = {
            'flow_upstream': self.flow_upstream,
            'flow_downstream': self.flow_downstream,
            'occupancy': self.occupancy_upstream,
            'queue_length': self.queue_length_upstream,
            'speed_limit': self.current_speed_limit,
            'collisions': len(self.collisions),
            'simulation_time': current_time,
            'simulation_step': self.simulation_step
        }
        
        return observation, reward, done, False, info

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Close existing SUMO if running
        if self.is_sumo_initialized:
            self.close_sumo("Environment reset")
        
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
        self.start_sumo()
                
        raw_observation = np.array([
            self.default_speed_limit / 3.6,
            0.0, 0.0, 0.0, 0.0, 0.0,
            self.default_speed_limit
        ], dtype=np.float64)

        observation = self.preprocess_state(raw_observation)

        info = {
            'flow_upstream': 0, 'flow_downstream': 0, 'occupancy': 0,
            'queue_length': 0, 'speed_limit': self.default_speed_limit,
            'collisions': 0, 'simulation_time': 0, 'simulation_step': 0
        }
        
        return observation, info

    def close_sumo(self, reason):
        pass

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
        queue_penalty = min((self.queue_length_upstream / MAX_QUEUE_LENGTH)**2, 1.0) * 0.1
        
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
        """Balanced reward incorporating delay equity principles from recent research."""
        # Primary flow efficiency (30%)
        R_flow = min(self.flow_smoothed / MAX_FLOW, 1.0) * 0.3
        
        # Speed efficiency with target speed consideration (25%)
        target_speed = 100.0 / 3.6  # 100 km/h target
        speed_efficiency = 1.0 - abs(self.avg_speed_before - target_speed) / target_speed
        R_speed = max(0.0, speed_efficiency) * 0.25
        
        # Speed harmonization (20%)
        R_smooth = self._calculate_speed_smoothness() * 0.2
        
        # Queue equity penalty (15%) - prevent concentrated congestion
        queue_penalty = min(self.queue_length_upstream / MAX_QUEUE_LENGTH, 1.0) * 0.15
        
        # Safety component (10%)
        safety_reward = -abs(self.collisions_penalty) * 0.1
        
        return R_flow + R_speed + R_smooth - queue_penalty + safety_reward + invalid_action_penalty

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

    def preprocess_state(self, raw_state):
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
        queue_length = np.clip(raw_state[3], 0, MAX_QUEUE_LENGTH) / MAX_QUEUE_LENGTH
        
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

    def apply_vsl_enforcement(self, speed_limit_kmh):
        """
        Apply Variable Speed Limit enforcement based on configured mode.
        
        Args:
            speed_limit_kmh (float): Speed limit in km/h
        """
        speed_limit_ms = speed_limit_kmh / 3.6  # Convert to m/s
        
        if self.vsl_enforcement == "lane_only":
            # Option 3: Only set maximum allowed speed for the lane
            for segId in seg_1_before:
                traci.lane.setMaxSpeed(segId, speed_limit_ms)
            logging.debug(f"VSL Mode 3: Set lane max speed to {speed_limit_kmh} km/h")
            
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
                        logging.debug(f"Could not set speed for vehicle {veh_id}: {e}")
            
            logging.debug(f"VSL Mode 1: Forced all vehicles to {speed_limit_kmh} km/h")
            
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
                        logging.debug(f"Could not check/set speed for vehicle {veh_id}: {e}")
            
            logging.debug(f"VSL Mode 2: Forced electric_passenger vehicles to {speed_limit_kmh} km/h")
            
        else:
            logging.warning(f"Unknown VSL enforcement mode: {self.vsl_enforcement}. Using lane_only.")
            # Fallback to lane_only
            for segId in seg_1_before:
                traci.lane.setMaxSpeed(segId, speed_limit_ms)

    def close(self):
        self.close_sumo("env.close()")

class TrafficDataLogger:
    """
    Comprehensive data logger for traffic simulation and RL training.
    Designed for SUMO-based VSL control experiments with SB3 integration.
    """
    
    def __init__(self, default_speed_limit=130):
        """
        Initialize the traffic data logger.
        
        Args:
            default_speed_limit (int): Default speed limit for the simulation (km/h)
        """
        self.default_speed_limit = default_speed_limit
        self.data = []
        self.step_count = 0
        self.episode_count = 0
        self.start_time = datetime.now()
        
        # Performance tracking variables
        self.total_reward = 0.0
        self.episode_rewards = []
        self.best_reward = float('-inf')
        self.collision_count = 0
        
        # Traffic metrics tracking
        self.total_vehicles_processed = 0
        self.avg_flow_rate = 0.0
        self.avg_occupancy = 0.0
        self.avg_queue_length = 0.0
        self.speed_limit_changes = 0
        self.last_speed_limit = default_speed_limit
        
        # Create output directory if it doesn't exist
        self.output_dir = Path("./logs/traffic_data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"TrafficDataLogger initialized with default speed limit: {default_speed_limit} km/h")

    def log_step_data(self, simulation_time, current_speed_limit, flow_upstream, 
                     flow_downstream, occupancy, queue_length, reward, action):
        """
        Log data for a single simulation step.
        
        Args:
            simulation_time (float): Current simulation time in seconds
            current_speed_limit (int): Applied speed limit in km/h
            flow_upstream (float): Upstream traffic flow (vehicles/hour)
            flow_downstream (float): Downstream traffic flow (vehicles/hour)
            occupancy (float): Detector occupancy percentage
            queue_length (float): Queue length in meters
            reward (float): Reward received for this step
            action (int): Action taken (0: -5km/h, 1: 0km/h, 2: +5km/h)
        """
        # Track speed limit changes for control smoothness analysis[2]
        if current_speed_limit != self.last_speed_limit:
            self.speed_limit_changes += 1
            self.last_speed_limit = current_speed_limit
        
        # Convert action to human-readable format
        action_map = {0: -5, 1: 0, 2: 5}
        if isinstance(action, np.ndarray):
            action_scalar = action.item() if action.size == 1 else action[0]
        else:
            action_scalar = action
        speed_change = action_map.get(action_scalar, 0)
        
        # Calculate derived metrics
        flow_efficiency = (flow_downstream / max(flow_upstream, 1)) * 100  # Percentage
        capacity_utilization = (flow_downstream / 7200) * 100  # Assuming max capacity 7200 veh/h
        
        step_data = {
            'timestamp': datetime.now().isoformat(),
            'simulation_time': simulation_time,
            'step': self.step_count,
            'episode': self.episode_count,
            'current_speed_limit': current_speed_limit,
            'speed_change': speed_change,
            'action': action,
            'flow_upstream': flow_upstream,
            'flow_downstream': flow_downstream,
            'flow_efficiency': flow_efficiency,
            'capacity_utilization': capacity_utilization,
            'occupancy': occupancy,
            'queue_length': queue_length,
            'reward': reward,
            'cumulative_reward': self.total_reward + reward,
            'speed_limit_changes_total': self.speed_limit_changes
        }
        
        self.data.append(step_data)
        self.step_count += 1
        self.total_reward += reward
        
        # Update running averages for performance tracking
        self._update_running_averages(flow_downstream, occupancy, queue_length)
        
        # Log significant events
        if abs(speed_change) > 0:
            logging.debug(f"Speed limit changed by {speed_change} km/h to {current_speed_limit} km/h at step {self.step_count}")
        
        if reward < -10:
            logging.warning(f"Large negative reward ({reward:.2f}) at step {self.step_count}")

    def log_episode_end(self, episode_reward, episode_length, final_metrics=None):
        """
        Log episode completion data.
        
        Args:
            episode_reward (float): Total reward for the episode
            episode_length (int): Number of steps in the episode
            final_metrics (dict, optional): Additional episode metrics
        """
        self.episode_rewards.append(episode_reward)
        self.episode_count += 1
        
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            logging.info(f"New best episode reward: {episode_reward:.2f}")
        
        episode_data = {
            'episode': self.episode_count,
            'episode_reward': episode_reward,
            'episode_length': episode_length,
            'avg_reward_per_step': episode_reward / max(episode_length, 1),
            'speed_limit_changes': self.speed_limit_changes,
            'avg_flow_rate': self.avg_flow_rate,
            'avg_occupancy': self.avg_occupancy,
            'avg_queue_length': self.avg_queue_length,
            'timestamp': datetime.now().isoformat()
        }
        
        if final_metrics:
            episode_data.update(final_metrics)
        
        logging.info(f"Episode {self.episode_count} completed: "
                    f"Reward={episode_reward:.2f}, Length={episode_length}, "
                    f"Avg Flow={self.avg_flow_rate:.1f} veh/h")
        
        # Reset episode-specific counters
        self.speed_limit_changes = 0
        self.last_speed_limit = self.default_speed_limit

    def _update_running_averages(self, flow_downstream, occupancy, queue_length):
        """Update running averages for key traffic metrics."""
        alpha = 0.1  # Exponential moving average factor
        
        if self.step_count == 1:
            # Initialize with first values
            self.avg_flow_rate = flow_downstream
            self.avg_occupancy = occupancy
            self.avg_queue_length = queue_length
        else:
            # Update exponential moving averages
            self.avg_flow_rate = (1 - alpha) * self.avg_flow_rate + alpha * flow_downstream
            self.avg_occupancy = (1 - alpha) * self.avg_occupancy + alpha * occupancy
            self.avg_queue_length = (1 - alpha) * self.avg_queue_length + alpha * queue_length

    def save_to_csv(self, filename=None, include_summary=True):
        """
        Save logged data to CSV file with optional performance summary.
        
        Args:
            filename (str, optional): Custom filename. If None, auto-generates based on timestamp
            include_summary (bool): Whether to include summary statistics
        """
        if not self.data:
            logging.warning("No data to save")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"traffic_simulation_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        try:
            # Save step-by-step data[3]
            df = pd.DataFrame(self.data)
            df.to_csv(filepath, index=False)
            
            # Save summary statistics if requested
            if include_summary:
                summary_filepath = filepath.with_suffix('.summary.csv')
                self._save_summary_statistics(summary_filepath)
            
            logging.info(f"Traffic data saved to {filepath}")
            logging.info(f"Total steps logged: {len(self.data)}")
            
        except Exception as e:
            logging.error(f"Error saving data to {filepath}: {e}")

    def _save_summary_statistics(self, filepath):
        """Save summary statistics to separate file."""
        if not self.data or not self.episode_rewards:
            return
        
        df = pd.DataFrame(self.data)
        
        summary_stats = {
            'training_duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            'total_episodes': self.episode_count,
            'total_steps': len(self.data),
            'avg_episode_length': len(self.data) / max(self.episode_count, 1),
            'best_episode_reward': self.best_reward,
            'avg_episode_reward': np.mean(self.episode_rewards),
            'std_episode_reward': np.std(self.episode_rewards),
            'total_speed_limit_changes': df['speed_limit_changes_total'].iloc[-1] if len(df) > 0 else 0,
            'avg_flow_downstream': df['flow_downstream'].mean(),
            'max_flow_downstream': df['flow_downstream'].max(),
            'avg_occupancy': df['occupancy'].mean(),
            'max_queue_length': df['queue_length'].max(),
            'avg_reward_per_step': df['reward'].mean(),
            'min_reward': df['reward'].min(),
            'max_reward': df['reward'].max(),
            'action_distribution_decrease': (df['action'] == 0).sum(),
            'action_distribution_maintain': (df['action'] == 1).sum(),
            'action_distribution_increase': (df['action'] == 2).sum(),
            'default_speed_limit': self.default_speed_limit
        }
        
        # Calculate control smoothness metrics
        speed_changes = df['speed_change'].abs()
        summary_stats.update({
            'control_smoothness_avg_change': speed_changes.mean(),
            'control_smoothness_max_change': speed_changes.max(),
            'control_smoothness_std': speed_changes.std()
        })
        
        # Save summary
        summary_df = pd.DataFrame([summary_stats])
        summary_df.to_csv(filepath, index=False)
        
        logging.info(f"Summary statistics saved to {filepath}")

    def get_performance_metrics(self):
        """
        Get current performance metrics for monitoring during training[2].
        
        Returns:
            dict: Dictionary containing key performance indicators
        """
        if not self.data:
            return {}
        
        df = pd.DataFrame(self.data)
        
        return {
            'total_steps': len(self.data),
            'total_episodes': self.episode_count,
            'current_avg_reward': np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards),
            'best_reward': self.best_reward,
            'avg_flow_rate': self.avg_flow_rate,
            'avg_occupancy': self.avg_occupancy,
            'avg_queue_length': self.avg_queue_length,
            'recent_reward_trend': np.mean(df['reward'].tail(50)) if len(df) >= 50 else np.mean(df['reward']),
            'speed_limit_changes_rate': self.speed_limit_changes / max(len(df), 1),
            'training_time_minutes': (datetime.now() - self.start_time).total_seconds() / 60
        }

    def reset_episode_data(self):
        """Reset episode-specific data while keeping historical records."""
        self.speed_limit_changes = 0
        self.last_speed_limit = self.default_speed_limit
        self.total_reward = 0.0

    def export_for_analysis(self, export_format='pandas'):
        """
        Export data in various formats for external analysis.
        
        Args:
            export_format (str): Format for export ('pandas', 'numpy', 'dict')
            
        Returns:
            Data in requested format
        """
        if not self.data:
            return None
        
        if export_format == 'pandas':
            return pd.DataFrame(self.data)
        elif export_format == 'numpy':
            df = pd.DataFrame(self.data)
            return df.select_dtypes(include=[np.number]).values
        elif export_format == 'dict':
            return self.data.copy()
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

    def __len__(self):
        """Return number of logged steps."""
        return len(self.data)

    def __str__(self):
        """String representation of logger status."""
        return (f"TrafficDataLogger(steps={len(self.data)}, episodes={self.episode_count}, "
                f"avg_reward={np.mean(self.episode_rewards) if self.episode_rewards else 0:.2f})")

class TensorboardCallback(BaseCallback):
    def __init__(self, env, model, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.env = env  # Store the environment
        self.model = model  # Store the model

    def _on_step(self) -> bool:
        # Access metrics from the environment
        reward = self.locals.get('rewards', 0)  # Safeguard against missing keys
        # Assuming emissions_over_time, mean_speed_over_time, and flows are maintained in TrafficEnv
        # You might need to adjust based on actual implementation
        emissions = getattr(self.env, 'emissions_over_time', [0])[-1]
        mean_speed = getattr(self.env, 'mean_speed_over_time', [0])[-1]
        flow = getattr(self.env, 'flows', [0])[-1]
        
        # Record these values in TensorBoard using SB3's built-in logger
        self.logger.record('test/reward', reward)
        self.logger.record('test/emissions', emissions)
        self.logger.record('test/mean_speed', mean_speed)
        self.logger.record('test/flow', flow)
        
        return True  # Continue running the environment

class TrafficEnvForTuning(TrafficEnv):
    """
    Specialized TrafficEnv for hyperparameter tuning that uses pre-generated flow files.
    Inherits from TrafficEnv but skips flow generation to use scenario-specific files.
    """
    
    def __init__(self, port, model_name, model_idx, op_mode, base_gen_car_distrib, 
                 num_of_episodes=0, reward_fn="balanced", skip_flow_generation=True, vsl_enforcement="lane_only",
                 sumo_binary_path_override=None):
        super().__init__(port, model_name, model_idx, op_mode, base_gen_car_distrib, 
                        num_of_episodes, reward_fn, vsl_enforcement, sumo_binary_path_override)
        
        self.skip_flow_generation = skip_flow_generation # This is the primary flag
        
        if self.operation_mode == "train": # This is "tuning" mode
            self.sim_length = HYPER_PARAM_SIM_LENGTH

        # Override customization attributes from parent for tuning context
        self._sumo_start_context_prefix = "Tuning "
        # If sumo_binary_path_override is provided, it's used. Otherwise, tuning defaults to no-GUI.
        self._default_sumo_binary_for_env = os.path.join(os.environ['SUMO_HOME'], 'bin', sumoExecutable_nogui)
        # Tuning uses a different (potentially shorter) retry sleep logic
        self._sumo_retry_sleep_func = lambda attempt, max_retries_param_ignored: 1 + attempt 

        if self.skip_flow_generation:
            self._verify_flow_files()
    
    def _verify_flow_files(self):
        """Verify that required pre-generated flow files exist."""
        expected_flow_file = f"./traffic_environment/sumo/generated_flows_{self.model_name}_{self.model_idx}.rou.xml"
        expected_config_file = f"./traffic_environment/sumo/3_2_merge_{self.model_name}_{self.model_idx}.sumocfg"
        
        if not os.path.exists(expected_flow_file):
            logging.error(f"Missing pre-generated flow file: {expected_flow_file}")
            raise FileNotFoundError(f"Pre-generated flow file not found: {expected_flow_file}")
        
        if not os.path.exists(expected_config_file):
            logging.error(f"Missing pre-generated config file: {expected_config_file}")
            raise FileNotFoundError(f"Pre-generated config file not found: {expected_config_file}")
        
        logging.debug(f"Verified pre-generated files for scenario {self.model_idx}")
    
    def start_sumo(self):
        """
        Modified SUMO startup that optionally skips flow generation.
        Uses pre-generated scenario-specific flow files for consistent tuning.
        Relies on parent's start_sumo with overridden attributes.
        """
        if self.skip_flow_generation:
            # Log specific message for tuning if using pre-generated files
            logging.debug(f"Using pre-generated flow file for {self._get_sumo_log_identifier()}")
        
        # All other logic is now handled by the parent's start_sumo
        # using the overridden attributes (_sumo_start_context_prefix, 
        # _default_sumo_binary_for_env, _sumo_retry_sleep_func)
        super().start_sumo()
    
    def step(self, action):
        """
        Modified step function optimized for hyperparameter tuning.
        Maintains all functionality but with optimized logging and faster termination.
        """
        # Call parent step method
        observation, reward, done, truncated, info = super().step(action)
                
        # **TUNING OPTIMIZATION**: Early termination for clearly poor performers
        if hasattr(self, '_tuning_step_count'):
            self._tuning_step_count += 1
        else:
            self._tuning_step_count = 1
        
        # Early termination if performance is clearly poor after 50 steps
        if self._tuning_step_count > 50:
            if hasattr(self, '_cumulative_reward'):
                self._cumulative_reward += reward
            else:
                self._cumulative_reward = reward
            
            # If average reward is very negative, terminate early
            avg_reward = self._cumulative_reward / self._tuning_step_count
            if avg_reward < -5:  # Threshold for clearly poor performance
                logging.debug(f"Early termination for poor performance: avg_reward={avg_reward:.2f}")
                done = True
        else:
            if hasattr(self, '_cumulative_reward'):
                self._cumulative_reward += reward
            else:
                self._cumulative_reward = reward
        
        return observation, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset environment and tuning-specific counters."""
        # Reset tuning counters
        self._tuning_step_count = 0
        self._cumulative_reward = 0.0
        
        # Call parent reset
        return super().reset(seed, options)
    
    def close_sumo(self, reason):
        pass
    
    def get_tuning_metrics(self):
        """
        Get metrics specifically useful for hyperparameter tuning.
        
        Returns:
            dict: Tuning-relevant metrics
        """
        metrics = {
            'steps_completed': getattr(self, '_tuning_step_count', 0),
            'cumulative_reward': getattr(self, '_cumulative_reward', 0.0),
            'avg_reward': getattr(self, '_cumulative_reward', 0.0) / max(getattr(self, '_tuning_step_count', 1), 1),
            'current_flow': self.flow_downstream,
            'current_occupancy': self.occupancy_upstream,
            'queue_length': self.queue_length_upstream,
            'speed_limit': self.current_speed_limit,
            'scenario_id': self.model_idx
        }
        return metrics

    def close(self):
        self.close_sumo("env.close()")

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Main entry point for running the DRL VSL environment with SUMO. """

def worker(args, results_list):
    res = run_training_for_combination(args)
    results_list.append(res)

if __name__ == '__main__':
    # Suppress matplotlib debug output
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    TrafficEnv.close_sumo = _robust_close_sumo
    TrafficEnvForTuning.close_sumo = _robust_close_sumo

    # from https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        logging.info("SUMO environment is not set up correctly.")

    # VSL Enforcement Mode Selection
    # Options: "all_vehicles", "electric_only", "lane_only"
    vsl_mode = "electric_only"  # Change this to test different modes
    
    option = 4
    algo_used = "DQN"
    reward_used = "balanced"
    
    config_model_name = f"{algo_used}_{reward_used}_{vsl_mode}"  # Updated line

    if option == 1:
        create_sumocfg(config_model_name, vsl_mode)  # Add vsl_mode parameter
        train_model(algorithm=algo_used, 
                    reward_function=reward_used, 
                    use_enhanced_params=True,
                    vsl_enforcement=vsl_mode)
    elif option == 2:
        optimal_params = get_optimal_params(algorithm=algo_used, traffic_density="high", episode_length="long")
        create_sumocfg(config_model_name, vsl_mode)  # Add vsl_mode parameter
        train_model(algorithm=algo_used, 
                    reward_function=reward_used,
                    use_enhanced_params=False,
                    custom_params=optimal_params,
                    vsl_enforcement=vsl_mode)
    elif option == 3:
        # Update tuning function to also support VSL enforcement
        best_params = tune_hyperparameters(algorithm=algo_used, 
                                           reward_function=reward_used, 
                                           n_trials=15,
                                           vsl_enforcement=vsl_mode)
        create_sumocfg(config_model_name, vsl_mode)  # Add vsl_mode parameter
        if "net_arch_str" in best_params:
            net_arch_list = [int(x) for x in best_params["net_arch_str"].split(",")]
            best_params["net_arch"] = net_arch_list
            del best_params["net_arch_str"]

        train_model(algorithm=algo_used, 
                    reward_function=reward_used,
                    use_enhanced_params=False,
                    custom_params=best_params,
                    vsl_enforcement=vsl_mode)
    elif option == 4: 
        # Run all 9 combinations in parallel
        logging.info("Starting parallel training for 9 combinations.")
        
        # Use non-GUI SUMO for parallel runs
        parallel_sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', sumoExecutable_gui)

        reward_functions = ["mobility", "safety", "balanced"]
        vsl_enforcements = ["all_vehicles", "electric_only", "lane_only"]
        algo_to_use = "DQN" # Fixed as per current structure

        all_combinations_params = []
        process_counter = 0
        for r_fn in reward_functions:
            for vsl_m in vsl_enforcements:
                all_combinations_params.append((r_fn, vsl_m, process_counter, algo_to_use, parallel_sumo_binary))
                process_counter += 1
        
        # Determine number of parallel processes
        # User mentioned HW ability for 9, but let's be safe or allow configuration
        num_parallel_processes = min(9, mp.cpu_count() -1 if mp.cpu_count() > 1 else 1) 
        # num_parallel_processes = 3 # Or set to a fixed number like 3 if 9 is too much
        logging.info(f"Running {len(all_combinations_params)} combinations using {num_parallel_processes} parallel processes.")

        # Important for CUDA and multiprocessing on some OS
        if sys.platform.startswith("win") or sys.platform.startswith("darwin"): # Windows or macOS
             mp.set_start_method('spawn', force=True)

        processes = []
        results = mp.Manager().list()

        for args in all_combinations_params:
            p = mp.Process(target=worker, args=(args, results))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        logging.info("Parallel training run finished. Results:")
        for res in results:
            logging.info(res)
    # Evaluate the trained model
    # test_model(algorithm=algo_used, reward_function=reward_used, vsl_enforcement=vsl_mode)  # Add vsl_mode parameter
"""
Limitations and Future Work:
- Accepted error: "ERROR - Failed to remove ./traffic_environment/sumo\generated_flows_DQN_tune_102.rou.xml after 5 seconds."
- 
"""