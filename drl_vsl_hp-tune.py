# drl_vsl_hp-tune.py (FINAL REVISED VERSION WITH TRUE PARALLELISM)
import os
import sys
import time
import json
import glob
from pathlib import Path
from itertools import product
import optuna
from stable_baselines3 import DQN
import torch.nn as nn
import traci
import numpy as np
from typing import Optional
from datetime import datetime, timezone
from stable_baselines3.common.env_util import make_vec_env
import multiprocessing as mp

# Import your TrafficEnv and other shared code
from drl_vsl import (
    TrafficEnv, SUMO_CFG_TEMPLATE, BASE_TRAIN_SUMO_PORT,
    OPTUNA_PARAMS_DIR, sumoExecutable_gui, sumoExecutable_nogui, flow_generation_fix_num_veh, logger,
    SUMO_CONFIG_DIR, logging
)
from flow_gen import flow_generation, bimodal_distribution_24h

SUMO_EXE_GUI = sumoExecutable_nogui

# --- 1. TUNING CONFIGURATION ---
N_OPTUNA_TRIALS = 40
# *** KEY CHANGE FOR PARALLELISM ***
# We let the outer multiprocessing Pool handle the parallelism across combinations.
# Each individual Optuna study will run its trials sequentially (n_jobs=1) to avoid
# CPU over-subscription (e.g., 6 processes * 6 jobs = 36 jobs).
N_JOBS_PER_STUDY = 1

NUM_CANDIDATES_TO_VALIDATE = 3
N_VALIDATION_SEEDS = 3
BROAD_EXPLORATION_STEPS_PER_SCENARIO = 60
DEEP_VALIDATION_STEPS_PER_SCENARIO = 200

SHARED_DEMAND_SCENARIOS = [
    {"id": 100, "demand": 3500, "pattern": "uniform"},
    {"id": 101, "demand": 4500, "pattern": "uniform"},
    {"id": 102, "demand": 5500, "pattern": "uniform"},
    {"id": 103, "demand": 3000, "pattern": "bimodal"},
    {"id": 104, "demand": 4000, "pattern": "bimodal"},
]

def setup_worker_logging():
    """Configures logging for each worker process in the pool."""
    # Get the root logger used by your drl_vsl.py logger
    worker_logger = logging.getLogger() 
    
    # Set the level (e.g., INFO to see progress messages)
    worker_logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    if worker_logger.hasHandlers():
        worker_logger.handlers.clear()
        
    # Add a handler that prints to the console (stderr or stdout)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    worker_logger.addHandler(handler)

def suggest_hyperparameters(trial: optuna.Trial) -> dict:
    net_arch_str = trial.suggest_categorical("net_arch_str", ["128,128", "256,128", "256,256", "512,256,128"])
    hyperparams = {
        "net_arch_str": net_arch_str,
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "buffer_size": trial.suggest_categorical("buffer_size", [50000, 150000, 250000]),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "target_update_interval": trial.suggest_int("target_update_interval", 1000, 10000, log=True),
        "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.5),
        "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.05),
        "gamma": trial.suggest_float("gamma", 0.98, 0.999, log=True),
        "train_freq": trial.suggest_categorical("train_freq", [1, 2, 4]),
    }
    return hyperparams

def objective(trial: optuna.Trial,
              reward_fn: str, 
              vsl_enforcement: str,
              combination_name: str, 
              timesteps_per_scenario: int,
              base_port: int,
              is_validation: bool = False,
              fixed_params: Optional[dict] = None) -> float:
    
    if is_validation:
        hyperparams = fixed_params
    else:
        hyperparams = suggest_hyperparameters(trial)

    net_arch = [int(x.strip()) for x in hyperparams["net_arch_str"].split(',')]
    policy_kwargs = {"net_arch": net_arch, "activation_fn": nn.ReLU}
    
    model_params = hyperparams.copy()
    del model_params["net_arch_str"]
    if isinstance(model_params["train_freq"], int):
        model_params["train_freq"] = (model_params["train_freq"], "step")
    elif isinstance(model_params["train_freq"], list):
        model_params["train_freq"] = tuple(model_params["train_freq"])

    total_performance_score = 0.0
    
    for i, scenario_config in enumerate(SHARED_DEMAND_SCENARIOS):
        env = None
        try:
            port = base_port + (trial.number % 40) * len(SHARED_DEMAND_SCENARIOS) + i
            
            normalization_bounds_path = OPTUNA_PARAMS_DIR / f"best_optuna_hyperparams_{combination_name}.json"

            env_kwargs = dict(
                port=port,
                model_name=combination_name,
                model_idx=scenario_config["id"],
                sim_length=(timesteps_per_scenario + 10) * 60,
                base_gen_car_distrib=[scenario_config["pattern"], scenario_config["demand"]],
                num_of_episodes=1,
                reward_fn=reward_fn, 
                vsl_enforcement=vsl_enforcement,
                sumo_binary_path_override=SUMO_EXE_GUI,
                normalization_bounds_path=normalization_bounds_path
            )

            env = make_vec_env(TrafficEnv, n_envs=1, env_kwargs=env_kwargs)
            env.envs[0].skip_flow_generation = True

            model = DQN("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs, **model_params, device='cuda')
            
            model.learn(total_timesteps=timesteps_per_scenario, progress_bar=False)
            
            obs = env.reset()
            cumulative_reward = 0
            steps = 0
            done = np.array([False])
            while not done.any() and steps < timesteps_per_scenario:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                cumulative_reward += reward[0]
                steps += 1

            total_performance_score += cumulative_reward

            if not is_validation:
                trial.report(total_performance_score / (i + 1), i)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        except optuna.exceptions.TrialPruned as e:
            if env: env.close()
            raise e
        except Exception as e:
            logger.error(f"Trial {trial.number}, Scenario {i} for {combination_name} failed: {e}", exc_info=False)
            if env: env.close()
            return -1e9
        finally:
            if env:
                env.close()

    return total_performance_score / len(SHARED_DEMAND_SCENARIOS)

def save_best_params(best_trial: optuna.trial.FrozenTrial, output_path: str, algo: str):
    logger.info(f"Formatting and saving best parameters to {output_path}...")
    
    try:
        with open(output_path, "r") as f:
            existing_data = json.load(f)
        bounds_data = existing_data.get("bounds", {}) # Preserve the bounds dict
    except Exception as e:
        logger.warning(f"Could not read existing bounds from {output_path}: {e}. Bounds will not be saved.")
        bounds_data = {}
    
    hyperparams = best_trial.params
    net_arch = [int(x.strip()) for x in hyperparams.pop("net_arch_str").split(',')]
    train_freq_val = hyperparams["train_freq"]
    if isinstance(train_freq_val, int):
        train_freq_tuple = (train_freq_val, "step")
    else:
        train_freq_tuple = tuple(train_freq_val)
    
    formatted_params = {
        "learning_rate": hyperparams["learning_rate"],
        "buffer_size": hyperparams["buffer_size"],
        "batch_size": hyperparams["batch_size"],
        "target_update_interval": hyperparams["target_update_interval"],
        "exploration_fraction": hyperparams["exploration_fraction"],
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": hyperparams["exploration_final_eps"],
        "learning_starts": 10000,
        "train_freq": train_freq_tuple,
        "gradient_steps": 1,
        "tau": 1.0,
        "gamma": hyperparams["gamma"],
        "policy_kwargs": {"net_arch": net_arch, "activation_fn": "nn.ReLU"}
    }
    
    final_json = {
        algo: formatted_params,
        "bounds": bounds_data 
    }

    try:
        with open(output_path, "w") as f:
            json.dump(final_json, f, indent=4)
        logger.info(f"Successfully updated hyperparameters in: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save final hyperparameters to {output_path}: {e}")

def create_initial_hyperparameter_file(output_path: Path, combination_name: str, process_id: int):
    """Creates a placeholder hyperparameter file with default bounds if it does not exist."""
    if not output_path.exists():
        logger.info(f"[Process {process_id}] Placeholder not found. Creating default for {combination_name} at: {output_path}")
        initial_data = {
            "DQN": {
                 "policy_kwargs": {"net_arch": [256, 128], "activation_fn": "nn.ReLU"},
                 "learning_rate": 1e-4, "gamma": 0.99, "batch_size": 128, "buffer_size": 100000, 
                 "learning_starts": 10000, "train_freq": (4, "step"), "gradient_steps": 1, 
                 "target_update_interval": 5000, "tau": 1.0, "exploration_fraction": 0.2, 
                 "exploration_final_eps": 0.02
            },
            "bounds": {
                "max_flow": 8000.0,
                "max_occupancy": 100.0,
                "max_queue_length": 1500.0
            }
        }
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(initial_data, f, indent=4)
        except Exception as e:
            logger.error(f"[Process {process_id}] FAILED to create placeholder file: {e}")
            raise e # Raise the exception to stop this worker

def run_tuning_for_one_combination(args):
    """
    Worker function that encapsulates the entire tuning process for one combination.
    This function will be run in parallel by multiprocessing.Pool.
    """
    algo_to_tune, r_fn, vsl_m, process_id = args
    combination_name = f"{algo_to_tune}_{r_fn}_{vsl_m}"
    
    # Each process gets a dedicated block of ports to avoid collisions
    process_base_port = BASE_TRAIN_SUMO_PORT + process_id * 1000

    placeholder_path = OPTUNA_PARAMS_DIR / f"best_optuna_hyperparams_{combination_name}.json"
    create_initial_hyperparameter_file(placeholder_path, combination_name, process_id)

    logger.info("\n" + "="*80)
    logger.info(f"[Process {process_id}] STARTING TUNING FOR: {combination_name} on Port Base {process_base_port}")
    logger.info("="*80)

    run_timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    tuning_files_name = f"{combination_name}_tune_{process_id}_{run_timestamp}"

    # Pre-generate SUMO files
    for sc_cfg in SHARED_DEMAND_SCENARIOS:
        sim_len_sec = (DEEP_VALIDATION_STEPS_PER_SCENARIO + 10) * 60
        if sc_cfg["pattern"] == 'uniform':
            flow_generation_fix_num_veh(combination_name, sc_cfg["id"], sc_cfg["demand"], sim_len_sec, 1, 1)
        else:
            bimodal_pattern = bimodal_distribution_24h(sc_cfg["demand"] / 1000.0)
            flow_generation(combination_name, sc_cfg["id"], bimodal_pattern, sim_len_sec)
        
        cfg_content = SUMO_CFG_TEMPLATE.format(file_postfix=tuning_files_name)
        cfg_filepath = SUMO_CONFIG_DIR / f"3_2_merge_{tuning_files_name}.sumocfg"
        with open(cfg_filepath, 'w') as f:
            f.write(cfg_content)

    # Stage 1: Broad Exploration
    db_filename = f"{combination_name}_study.db"
    study_db_path = "sqlite:///" + os.path.join(OPTUNA_PARAMS_DIR, db_filename)
    study_broad = optuna.create_study(
        study_name=f"{combination_name}_broad",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        storage=study_db_path,
        load_if_exists=True,
    )
    
    objective_broad = lambda trial: objective(
        trial, r_fn, vsl_m, combination_name, BROAD_EXPLORATION_STEPS_PER_SCENARIO, process_base_port
    )
    study_broad.optimize(objective_broad, n_trials=N_OPTUNA_TRIALS, n_jobs=1)
    
    top_candidates = study_broad.best_trials[:NUM_CANDIDATES_TO_VALIDATE]

    # Stage 2: Deep Validation
    validated_results = []
    for i, candidate_trial in enumerate(top_candidates):
        seed_scores = []
        for seed in range(N_VALIDATION_SEEDS):
            validation_study = optuna.create_study(direction="maximize")
            dummy_trial = validation_study.ask() # Dummy trial for API compatibility
            score = objective(
                dummy_trial, r_fn, vsl_m, tuning_files_name,
                DEEP_VALIDATION_STEPS_PER_SCENARIO, process_base_port,
                is_validation=True, fixed_params=candidate_trial.params
            )
            seed_scores.append(score)
        
        avg_score = np.mean(seed_scores)
        std_score = np.std(seed_scores)
        validated_results.append({"trial": candidate_trial, "mean_score": avg_score})
        logger.info(f"[Process {process_id}] Candidate {i+1} for {combination_name} validation: Mean Score={avg_score:.2f} +/- {std_score:.2f}")

    # Final Step: Save Best
    if not validated_results:
        best_overall_trial = study_broad.best_trial
    else:
        best_validated = max(validated_results, key=lambda x: x['mean_score'])
        best_overall_trial = best_validated['trial']

    output_file_path = f"{OPTUNA_PARAMS_DIR}/best_optuna_hyperparams_{combination_name}.json"
    save_best_params(best_overall_trial, output_file_path, algo_to_tune)
    
    # Cleanup
    for f in glob.glob(f"./traffic_environment/sumo/*{tuning_files_name}*"):
        try:
            os.remove(f)
        except Exception as e:
            logger.warning(f"Could not remove temp file {f}: {e}")

    logger.info(f"[Process {process_id}] FINISHED TUNING FOR: {combination_name}")


if __name__ == '__main__':
    algo_to_tune = "DQN"
    reward_functions_to_tune = ["mobility", "safety", "balanced"]
    vsl_enforcements_to_tune = ["recommend", "electric_only"]

    os.makedirs(OPTUNA_PARAMS_DIR, exist_ok=True)
    
    # Create a list of all combination arguments for our worker function
    tuning_tasks = []
    for i, (r_fn, vsl_m) in enumerate(product(reward_functions_to_tune, vsl_enforcements_to_tune)):
        tuning_tasks.append((algo_to_tune, r_fn, vsl_m, i))

    # Determine the number of parallel processes to run
    num_parallel_processes = min(len(tuning_tasks), max(1, os.cpu_count() - 1))
    logger.info(f"Starting hyperparameter tuning for {len(tuning_tasks)} combinations using {num_parallel_processes} parallel processes.")

    # Use a multiprocessing Pool to run all tuning tasks in parallel
    # 'spawn' is a safer start method for complex applications, especially on Windows/macOS
    if sys.platform.startswith("win") or sys.platform.startswith("darwin"):
        mp.set_start_method('spawn', force=True)
        
    with mp.Pool(processes=num_parallel_processes, initializer=setup_worker_logging) as pool:
        pool.map(run_tuning_for_one_combination, tuning_tasks)

    logger.info("\nAll hyperparameter tuning combinations are complete.")