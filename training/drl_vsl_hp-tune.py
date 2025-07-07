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
from legacy.drl_vsl import (
    TrafficEnv, SUMO_CFG_TEMPLATE, BASE_TRAIN_SUMO_PORT,
    OPTUNA_PARAMS_DIR, sumoExecutable_gui, sumoExecutable_nogui, flow_generation_fix_num_veh, logger,
    SUMO_CONFIG_DIR, logging
)
from traffic_environment.flow_gen import flow_generation, bimodal_distribution_24h

SUMO_EXE_GUI = sumoExecutable_nogui

# --- 1. TUNING CONFIGURATION ---
N_OPTUNA_TRIALS = 30
N_JOBS_PER_STUDY = 1
NUM_CANDIDATES_TO_VALIDATE = 3
N_VALIDATION_SEEDS = 3
BROAD_EXPLORATION_STEPS_PER_SCENARIO = 2000
DEEP_VALIDATION_STEPS_PER_SCENARIO = 3000

SHARED_DEMAND_SCENARIOS = [
    {"id": 100, "demand": 2500, "pattern": "uniform"},
    {"id": 101, "demand": 3000, "pattern": "uniform"},
    {"id": 102, "demand": 3000, "pattern": "bimodal"},
    {"id": 103, "demand": 3500, "pattern": "uniform"},
    {"id": 104, "demand": 4000, "pattern": "bimodal"},
    {"id": 105, "demand": 4500, "pattern": "bimodal"},
    {"id": 106, "demand": 5000, "pattern": "bimodal"},
]

HYPERPARAM_FOCUSED_RUN = True  # Set to True for a focused search space, False for full exploration
"""
The "Sensitive" Parameters (The Search Focus)
* "net_arch_str": trial.suggest_categorical(...)
    What it does: Searches over three distinct network sizes.
    Justification: The neural network's architecture dictates the agent's representational capacity. 
    A network that is too small ([128,128]) may underfit, failing to capture the complex, 
    non-linear dynamics of traffic flow. A network that is too large may be slow to train and prone to 
    overfitting on the limited data from a short run. The choice of architecture is a fundamental 
    trade-off between model capacity and learnability. Therefore, it is essential to include it in 
    the search as it is a primary determinant of final policy quality. This aligns with the universal 
    machine learning principle of model selection.
* "learning_rate": trial.suggest_float(..., log=True)
    What it does: Searches for the optimal step size for the Adam optimizer, on a logarithmic scale. 
    The range 5e-5 to 5e-4 is a focused "sweet spot."
    Justification: The learning rate is arguably the single most sensitive hyperparameter in training 
    deep neural networks [Goodfellow, Bengio, & Courville, 2016, "Deep Learning"]. If it's too high, 
    the training will be unstable and may diverge. If it's too low, learning will be prohibitively slow, 
    especially within a constrained budget. Searching on a logarithmic scale is critical, as a change 
    from 1e-4 to 2e-4 has a much larger impact than a change from 1e-3 to 1.1e-3. This focuses the search 
    on the most impactful orders of magnitude.
* "gamma": trial.suggest_categorical(...)
    What it does: Searches over three high-value discount factors.
    Justification: The discount factor, gamma, defines the agent's planning horizon. It determines how much 
    weight is given to future rewards versus immediate rewards. In traffic control, the objective is to prevent 
    future congestion, making it a "farsighted" problem. A gamma value close to 1 is theoretically necessary for 
    the agent to learn proactive, long-term strategies. However, values very close to 1 (e.g., 0.999) can increase 
    variance in the value estimates. Therefore, exploring this high-value range is critical to finding 
    the optimal balance between foresight and learning stability for this specific task.
The "Robust" Parameters (Fixed for Efficiency)
* "buffer_size": 100000
    Justification: The experience replay buffer is essential for breaking temporal correlations in the data. 
    While a larger buffer can provide more diverse samples, it also means that older, potentially off-policy 
    data persists longer. For a short run, a moderately sized buffer of 100,000 provides a good balance. 
    It is large enough to ensure sample diversity without being so large that the agent cannot fill it 
    with meaningful experiences within the trial's duration.
* "batch_size": 128
    Justification: The batch size controls the trade-off between the accuracy of the gradient estimate and 
    the speed of updates. A size of 128 is a widely used and robust default in DRL literature that provides 
    a stable gradient estimate without being computationally prohibitive.
* "target_update_interval": 5000
    Justification: This parameter determines the stability of the TD (Temporal-Difference) target. 
    An update interval of 5,000 steps is a standard default in many successful 
    DQN implementations (including Stable-Baselines3). It ensures the target network remains stable long enough 
    for the Q-network to learn towards it, preventing the "moving target" problem.
* "exploration_fraction": 0.15 and "exploration_final_eps": 0.05
    Justification: These parameters define a linear annealing schedule for the exploration rate (epsilon). 
    A schedule that anneals over 15% of the total timesteps to a final value of 5% exploration 
    is a standard configuration that ensures sufficient initial exploration to discover the 
    environment's dynamics, followed by a phase of exploitation to refine the learned policy.
* "learning_starts": 5000
    Justification: This parameter ensures the replay buffer is populated with a minimum number 
    of diverse experiences before learning begins. Starting to learn from a small or correlated 
    set of initial samples can lead to catastrophic forgetting or convergence to a poor local optimum. 
    A value of 5,000 is a safe minimum that ensures initial learning batches are representative.
"""
def suggest_hyperparameters_for_short_run(trial: optuna.Trial) -> dict:
    """
    An informed and focused search space for a budget-constrained hyperparameter search.

    To maximize the efficiency of the hyperparameter search within a limited computational budget, 
    this function is used as a strategy of focused parameter space design. 
    Based on a review of DRL applications in traffic control, the learning rate, network architecture, 
    and discount factor are identified as the most critical determinants of agent performance. 
    """
    
    # --- Define the fixed, "robust" parameters ---
    hyperparams = {
        "buffer_size": 100000,
        "batch_size": 128,
        "target_update_interval": 5000,
        "exploration_fraction": 0.15,
        "exploration_final_eps": 0.05,
        "learning_starts": 5000,
        "train_freq": 4, # This will be correctly converted to a tuple later
        "gradient_steps": 1,
    }

    # --- Suggest the "sensitive" parameters and update the dictionary ---
    hyperparams.update({
        "net_arch_str": trial.suggest_categorical("net_arch_str", ["128,128", "256,128", "512,256,128"]),
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True),
        "gamma": trial.suggest_categorical("gamma", [0.99, 0.995, 0.999]),
    })

    return hyperparams

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
              action_strategy: str,
              state_representation: str,
              vsl_enforcement: str,
              combination_name: str, 
              timesteps_per_scenario: int,
              base_port: int,
              is_validation: bool = False,
              fixed_params: Optional[dict] = None) -> float:
    
    if is_validation:
        hyperparams = {
            "buffer_size": 100000,
            "batch_size": 128,
            "target_update_interval": 5000,
            "exploration_fraction": 0.15,
            "exploration_final_eps": 0.05,
            "learning_starts": 5000,
            "train_freq": 4,
            "gradient_steps": 1,
        }
        hyperparams.update(fixed_params)
    else:
        if HYPERPARAM_FOCUSED_RUN:
            logger.info(f"Using focused hyperparameter search for trial {trial.number} in {combination_name}")
            hyperparams = suggest_hyperparameters_for_short_run(trial)
        else:
            logger.info(f"Using full hyperparameter search for trial {trial.number} in {combination_name}")
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
            port = base_port + (trial.number % 100) + i
            normalization_bounds_path = OPTUNA_PARAMS_DIR / f"best_optuna_hyperparams_{combination_name}.json"
            
            env_kwargs = dict(
                port=port,
                model_name=combination_name,
                model_idx=scenario_config["id"],
                sim_length=(timesteps_per_scenario + 10) * 60,
                base_gen_car_distrib=[scenario_config["pattern"], scenario_config["demand"]],
                num_of_episodes=1,
                reward_fn=reward_fn,
                action_strategy=action_strategy,
                state_representation=state_representation,
                vsl_enforcement=vsl_enforcement,
                sumo_binary_path_override=SUMO_EXE_GUI,
                normalization_bounds_path=normalization_bounds_path
            )

            env = make_vec_env(TrafficEnv, n_envs=1, env_kwargs=env_kwargs)
            env.envs[0].skip_flow_generation = True

            model = DQN("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs, **model_params, device='cuda')
            
            model.learn(total_timesteps=timesteps_per_scenario, progress_bar=True)
            
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

def save_best_params(proposed_trial: optuna.trial.FrozenTrial,
                     output_path: str,
                     algo: str,
                     proposed_mean_score: float,
                     run_timestamp: str):
    
    logger.info(f"Gatekeeper save function initiated for {output_path} with proposed score: {proposed_mean_score:.2f}")

    # --- 1. LOAD EXISTING DATA AND BEST HISTORICAL SCORE ---
    try:
        with open(output_path, "r") as f:
            existing_data = json.load(f)
        validation_history = existing_data.get("validation_history", [])
        if validation_history:
            best_historical_score = max(entry['mean_score'] for entry in validation_history)
        else:
            best_historical_score = -float('inf')
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}
        validation_history = []
        best_historical_score = -float('inf')

    logger.info(f"Best historical score: {best_historical_score:.2f}. New proposed score: {proposed_mean_score:.2f}")

    # --- 2. COMPARE AND DECIDE WHETHER TO UPDATE HYPERPARAMETERS ---
    if proposed_mean_score > best_historical_score:
        logger.info("New score is better! Updating the main hyperparameter block.")
        
        # --- THIS IS THE CRITICAL FIX ---
        # `proposed_trial.params` only contains the searched parameters.
        # We must reconstruct the full dictionary by combining it with our fixed values.
        
        # Start with the base of fixed parameters (must match suggest_hyperparameters_for_short_run)
        if HYPERPARAM_FOCUSED_RUN:
            full_hyperparams = {
                "buffer_size": 100000,
                "batch_size": 128,
                "target_update_interval": 5000,
                "exploration_fraction": 0.15,
                "exploration_final_eps": 0.05,
                "learning_starts": 5000,
                "train_freq": 4,
                "gradient_steps": 1,
            }
            # Update with the values that were actually searched
            full_hyperparams.update(proposed_trial.params)
        else:
            # If not a focused run, then trial.params should contain everything
            full_hyperparams = proposed_trial.params

        # --- Now, build the formatted_params dict using the complete 'full_hyperparams' ---
        net_arch = [int(x.strip()) for x in full_hyperparams.pop("net_arch_str").split(',')]
        
        # Use .get() for safety, though the keys should now exist
        train_freq_val = full_hyperparams.get("train_freq", 4)
        train_freq_tuple = (train_freq_val, "step") if isinstance(train_freq_val, int) else tuple(train_freq_val)

        # Build the final dictionary for the JSON file
        formatted_params = {
            "learning_rate": full_hyperparams["learning_rate"],
            "buffer_size": full_hyperparams["buffer_size"],
            "batch_size": full_hyperparams["batch_size"],
            "target_update_interval": full_hyperparams["target_update_interval"],
            "exploration_fraction": full_hyperparams["exploration_fraction"],
            "exploration_initial_eps": 1.0, # This is a static choice
            "exploration_final_eps": full_hyperparams["exploration_final_eps"],
            "learning_starts": full_hyperparams.get("learning_starts", 20000), # Use .get for safety
            "train_freq": train_freq_tuple,
            "gradient_steps": full_hyperparams.get("gradient_steps", 1),
            "tau": 1.0, # Static choice
            "gamma": full_hyperparams["gamma"],
            "policy_kwargs": {"net_arch": net_arch, "activation_fn": "nn.ReLU"}
        }
        existing_data[algo] = formatted_params
    else:
        # ... (this part is unchanged) ...
        logger.info("Proposed score is not better than historical best. Main hyperparameters will not be changed.")
        if algo not in existing_data:
            existing_data[algo] = {}

    # --- 3. APPEND THE CURRENT RUN TO THE HISTORY (ALWAYS) ---
    new_validation_entry = {
        "timestamp": run_timestamp,
        "mean_score": round(proposed_mean_score, 4),
        "params": proposed_trial.params
    }
    validation_history.append(new_validation_entry)
    validation_history.sort(key=lambda x: x["timestamp"])
    existing_data["validation_history"] = validation_history # Keep it chronological
    
    # --- 4. PRESERVE BOUNDS and SAVE THE FILE ---
    # Ensure the bounds dict is preserved or initialized
    if "bounds" not in existing_data:
        existing_data["bounds"] = {}

    final_json = {
        algo: existing_data[algo],
        "bounds": existing_data.get("bounds"),
        "validation_history": existing_data["validation_history"]
    }

    try:
        with open(output_path, "w") as f:
            json.dump(final_json, f, indent=4)
        logger.info(f"Successfully updated file: {output_path}")
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
                 "learning_starts": 20000, "train_freq": (4, "step"), "gradient_steps": 1, 
                 "target_update_interval": 2000, "tau": 1.0, "exploration_fraction": 0.2, 
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
    algo_to_tune, r_fn, vsl_m, process_id, cavs_perc = args
    combination_name = f"{algo_to_tune}_{r_fn}_{vsl_m}"
    # Each worker process gets a dedicated block of 1000 ports to avoid any collisions.
    process_base_port = BASE_TRAIN_SUMO_PORT + process_id * 1000
    placeholder_path = OPTUNA_PARAMS_DIR / f"best_optuna_hyperparams_{combination_name}.json"
    create_initial_hyperparameter_file(placeholder_path, combination_name, process_id)
    logger.info(f"\n[Worker {process_id}] STARTING TUNING FOR: {combination_name} on Port Base {process_base_port}")
    run_timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    # Determine the maximum simulation length needed to cover both stages
    max_sim_len_needed = (max(BROAD_EXPLORATION_STEPS_PER_SCENARIO, DEEP_VALIDATION_STEPS_PER_SCENARIO) + 10) * 60
    for sc_cfg in SHARED_DEMAND_SCENARIOS:
        if sc_cfg["pattern"] == 'uniform':
            flow_generation_fix_num_veh(combination_name, sc_cfg["demand"], max_sim_len_needed, 1, 1, cavs_perc)
        else:
            bimodal_pattern = bimodal_distribution_24h(sc_cfg["demand"] / 1000.0)
            flow_generation(combination_name, bimodal_pattern, max_sim_len_needed, cavs_perc)
    # Create the single .sumocfg file that points to these flows
    cfg_content = SUMO_CFG_TEMPLATE.format(file_postfix=combination_name)
    cfg_filepath = SUMO_CONFIG_DIR / f"3_2_merge_{combination_name}.sumocfg"
    with open(cfg_filepath, 'w') as f:
        f.write(cfg_content)
    logger.info(f"[Worker {process_id}] File generation complete.")

    # Stage 1: Broad Exploration
    db_filename = f"{combination_name}_study.db"
    study_db_path = "sqlite:///" + os.path.join(OPTUNA_PARAMS_DIR, db_filename)
    study_broad = optuna.create_study(
        study_name=f"{combination_name}_broad",
        direction="maximize",
        # pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
        pruner=optuna.pruners.PercentilePruner(percentile=25, n_warmup_steps=3), # [%] of trials are pruned and it will not prune before x [steps]
        storage=study_db_path,
        load_if_exists=True,
    )
    
    objective_broad = lambda trial: objective(
        trial, r_fn, vsl_m, "absolute_speed", "full_metrics", combination_name, BROAD_EXPLORATION_STEPS_PER_SCENARIO, process_base_port
    )
    study_broad.optimize(objective_broad, n_trials=N_OPTUNA_TRIALS, n_jobs=N_JOBS_PER_STUDY)
    logger.info(f"[Worker {process_id}] Broad search complete. Starting deep validation...")
    top_candidates = study_broad.best_trials[:NUM_CANDIDATES_TO_VALIDATE]
    
    # Stage 2: Deep Validation
    validated_results = []
    for i, candidate_trial in enumerate(top_candidates):
        seed_scores = []
        logger.info(f"[Worker {process_id}] Validating candidate {i+1}/{len(top_candidates)}...")
        for seed in range(N_VALIDATION_SEEDS):
            # Create a dummy trial for API compatibility
            validation_study = optuna.create_study(direction="maximize")
            dummy_trial = validation_study.ask() 
            
            # The objective function is called just like in Stage 1, but with different parameters.
            # It will correctly find and use the files for `combination_name`.
            score = objective(
                dummy_trial, r_fn, vsl_m, 
                "absolute_speed",
                "full_metrics",
                combination_name, # Pass the consistent name
                DEEP_VALIDATION_STEPS_PER_SCENARIO, process_base_port,
                is_validation=True, fixed_params=candidate_trial.params
            )
            seed_scores.append(score)
            logger.info(f"  - Seed {seed+1}/{N_VALIDATION_SEEDS} score: {score:.2f}")

        avg_score = np.mean(seed_scores)
        std_score = np.std(seed_scores)
        validated_results.append({"trial": candidate_trial, "mean_score": avg_score})
        logger.info(f"  -> Candidate {i+1} validation complete: Mean Score={avg_score:.2f} +/- {std_score:.2f}")

    final_mean_score = 0.0
    # Final Step: Save Best
    if not validated_results:
        best_overall_trial = study_broad.best_trial
        final_mean_score = best_overall_trial.value
    else:
        best_validated = max(validated_results, key=lambda x: x['mean_score'])
        best_overall_trial = best_validated['trial']
        final_mean_score = best_validated['mean_score']

    output_file_path = OPTUNA_PARAMS_DIR / f"best_optuna_hyperparams_{combination_name}.json"
    
    # This call now proposes the result to the gatekeeper function
    if best_overall_trial is not None:
        save_best_params(
            best_overall_trial,
            str(output_file_path),
            algo_to_tune,
            proposed_mean_score=final_mean_score,
            run_timestamp=run_timestamp
        )
    
    # --- Cleanup ---
    logger.info(f"[Worker {process_id}] Cleaning up files for {combination_name}...")
    for f in glob.glob(f"./traffic_environment/sumo/*{combination_name}*"):
        try:
            os.remove(f)
        except Exception as e:
            logger.warning(f"Could not remove temp file {f}: {e}")

    logger.info(f"[Worker {process_id}] FINISHED TUNING FOR: {combination_name}")

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

"""==============================================================================================="""
if __name__ == '__main__':
    algo_to_tune = "DQN"
    reward_functions_to_tune = ["mobility", "safety", "balanced"]
    vsl_enforcements_to_tune = ["recommend", "cavs_only"]
    CAVS_PERCENTAGE = 20

    os.makedirs(OPTUNA_PARAMS_DIR, exist_ok=True)
    
    # Create a list of all combination arguments for our worker function
    tuning_tasks = []
    for i, (r_fn, vsl_m) in enumerate(product(reward_functions_to_tune, vsl_enforcements_to_tune)):
        tuning_tasks.append((algo_to_tune, r_fn, vsl_m, i, CAVS_PERCENTAGE))

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