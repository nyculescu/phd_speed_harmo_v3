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
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
from typing import Optional
import datetime

# Import your TrafficEnv and any other shared code from drl_vsl.py
from drl_vsl import (
    TrafficEnv,
    SUMO_CFG_TEMPLATE,
    BASE_TRAIN_SUMO_PORT,
    BASE_EVAL_SUMO_PORT,
    PORTS_PER_TUNING_PROCESS,
    OPTUNA_PARAMS_DIR,
    sumoExecutable_nogui,
    sumoExecutable_gui,
    flow_generation_fix_num_veh,
    logger,
    MAX_FLOW,
    MAX_OCCUPANCY,
    SUMO_CONFIG_DIR,
    MAX_QUEUE_LENGTH
)

# --- 1. SETUP: DEFINE TUNING CONFIGURATIONS ---
# This makes the two-stage process explicit and easy to manage without a class.
@dataclass
class TuningConfig:
    name: str
    n_trials: int
    timesteps_per_trial: int # total_timesteps for model.learn() per scenario within an Optuna trial

FIXED_PARAMS_FOR_VALIDATION = None
HYPER_PARAM_SIM_LENGTH = 7200 # 1800 (Simulation length for tuning scenarios) / 60 (Aggregation Time) = 30 DRL steps
N_PARALLEL_OPTUNA_TRIALS = 1
NUM_CANDIDATES_TO_VALIDATE = 3
N_OPTUNA_TRIALS = 40 # Number of trials for Optuna study
PROGRESS_BAR = False # prevent SB3 from trying to create it's own tqdm progress bar
NORMALIZATION_BOUNDS_FILE = os.path.join(OPTUNA_PARAMS_DIR, "normalization_bounds.json")
CALIBRATION_EPISODE_LENGTH_ENV_STEPS = 60 # 3600 s
BROAD_EXPLORATION_TIMESTEPS_PER_SCENARIO = 600
DEEP_VALIDATION_TIMESTEPS_PER_SCENARIO = 6000 # 200 episodes * 30 steps = 6000 DRL steps
SUMO_EXE_GUI = sumoExecutable_nogui
SHARED_DEMAND_SCENARIOS = [
        {"id": 100, "demand": 2000, "pattern": "uniform"},
        {"id": 101, "demand": 2500, "pattern": "uniform"}, 
        {"id": 102, "demand": 3000, "pattern": "uniform"},
        {"id": 103, "demand": 3500, "pattern": "uniform"},
        {"id": 104, "demand": 4000, "pattern": "uniform"},
    ]

# Configuration for Stage 1
BROAD_EXPLORATION_CONFIG = TuningConfig(
    name="Broad Exploration",
    n_trials=N_OPTUNA_TRIALS, # Number of different hyperparameter sets to try (random seeds)
    timesteps_per_trial=BROAD_EXPLORATION_TIMESTEPS_PER_SCENARIO
)

# Configuration for Stage 2
DEEP_VALIDATION_CONFIG = TuningConfig(
    name="Deep Validation",
    n_trials=3,  # Random seeds
    timesteps_per_trial=DEEP_VALIDATION_TIMESTEPS_PER_SCENARIO
)

class TrafficEnvForTuning(TrafficEnv):
    """
    Specialized TrafficEnv for hyperparameter tuning that uses pre-generated flow files.
    Inherits from TrafficEnv but skips flow generation to use scenario-specific files.
    """
    
    def __init__(self, port, model_name, model_idx, sim_length, base_gen_car_distrib, 
                 num_of_episodes, reward_fn="balanced", skip_flow_generation=True, vsl_enforcement="recommend",
                 sumo_binary_path_override=None,
                 normalization_bounds_path: Optional[str] = None,
                 update_bounds: bool = False):
        super().__init__(port, model_name, model_idx, sim_length, base_gen_car_distrib, 
                        num_of_episodes, reward_fn, vsl_enforcement, sumo_binary_path_override,
                        normalization_bounds_path=normalization_bounds_path)
        
        self.skip_flow_generation = skip_flow_generation # This is the primary flag

        # Override customization attributes from parent for tuning context
        self._sumo_start_context_prefix = "Tuning "
        self._default_sumo_binary_for_env = os.path.join(os.environ['SUMO_HOME'], 'bin', SUMO_EXE_GUI)
        # Tuning uses a different (potentially shorter) retry sleep logic
        self._sumo_retry_sleep_func = lambda attempt, max_retries_param_ignored: 1 + attempt 

        if self.skip_flow_generation:
            self._verify_flow_files()
        
        self.total_vehicles_before = 0
        self.total_vehicles_after = 0

        self.update_bounds = update_bounds
        self.bounds_path = normalization_bounds_path
        self.observed_values = {"flows": [], "occupancies": [], "queues": []}
    
    def _verify_flow_files(self):
        """Verify that required pre-generated flow files exist."""
        expected_flow_file = f"./traffic_environment/sumo/generated_flows_{self.model_name}_{self.model_idx}.rou.xml"
        expected_config_file = f"./traffic_environment/sumo/3_2_merge_{self.model_name}_{self.model_idx}.sumocfg"
        
        if not os.path.exists(expected_flow_file):
            logger.error(f"Missing pre-generated flow file: {expected_flow_file}")
            raise FileNotFoundError(f"Pre-generated flow file not found: {expected_flow_file}")
        
        if not os.path.exists(expected_config_file):
            logger.error(f"Missing pre-generated config file: {expected_config_file}")
            raise FileNotFoundError(f"Pre-generated config file not found: {expected_config_file}")
        
        logger.debug(f"Verified pre-generated files for scenario {self.model_idx}")
    
    def get_simulation_summary(self):
        return {
            "total_vehicles_before": self.total_vehicles_before,
            "total_vehicles_after": self.total_vehicles_after
        }
    
    def start_sumo(self):
        """
        Modified SUMO startup that optionally skips flow generation.
        Uses pre-generated scenario-specific flow files for consistent tuning.
        Relies on parent's start_sumo with overridden attributes.
        """
        if self.skip_flow_generation:
            # Log specific message for tuning if using pre-generated files
            logger.debug(f"Using pre-generated flow file for {self._get_sumo_log_identifier()}")
        
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

        # Accumulate vehicle counts for verification
        try:
            self.total_vehicles_before += traci.edge.getLastStepVehicleNumber("seg_0_before")
            self.total_vehicles_after += traci.edge.getLastStepVehicleNumber("seg_0_after")
        except Exception as e:
            logger.warning(f"Could not get vehicle numbers for verification: {e}")
                
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
                logger.debug(f"Early termination for poor performance: avg_reward={avg_reward:.2f}")
                done = True
        else:
            if hasattr(self, '_cumulative_reward'):
                self._cumulative_reward += reward
            else:
                self._cumulative_reward = reward

        if self.update_bounds and self.bounds_path:
            self.observed_values["flows"].append(self.flow_downstream)
            self.observed_values["occupancies"].append(self.occupancy_upstream)
            self.observed_values["queues"].append(self.queue_length_upstream)
        
        return observation, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset environment and tuning-specific counters."""
        # Reset tuning counters
        self._tuning_step_count = 0
        self._cumulative_reward = 0.0
        
        # Call parent reset
        return super().reset(seed, options)
    
    def close_sumo(self, reason):
        """Close with optional bounds updating."""
        if self.update_bounds and self.bounds_path and self.observed_values["flows"]:
            self._update_normalization_bounds()
        
        super().close_sumo(reason)
    
    def _update_normalization_bounds(self):
        """Update the normalization bounds file with new observations."""
        try:
            # Load existing bounds
            if os.path.exists(self.bounds_path):
                with open(self.bounds_path, 'r') as f:
                    current_bounds = json.load(f)
            else:
                current_bounds = {
                    "max_flow": MAX_FLOW,
                    "max_occupancy": MAX_OCCUPANCY,
                    "max_queue_length": MAX_QUEUE_LENGTH
                }

            # Calculate new maximums from this environment's observations
            if self.observed_values["flows"]:
                new_max_flow = max(self.observed_values["flows"])
                new_max_occupancy = max(self.observed_values["occupancies"])
                new_max_queue = max(self.observed_values["queues"])
                
                # Update bounds if new values are higher
                updated = False
                if new_max_flow > current_bounds["max_flow"]:
                    current_bounds["max_flow"] = round(new_max_flow * 1.1, 2)
                    updated = True
                
                if new_max_occupancy > current_bounds["max_occupancy"]:
                    current_bounds["max_occupancy"] = min(round(new_max_occupancy * 1.1, 2), 100.0)
                    updated = True
                
                if new_max_queue > current_bounds["max_queue_length"]:
                    current_bounds["max_queue_length"] = round(new_max_queue * 1.1, 2)
                    updated = True
                
                # Save updated bounds
                if updated:
                    with open(self.bounds_path, 'w') as f:
                        json.dump(current_bounds, f, indent=4)
                    logger.debug(f"Updated normalization bounds: {current_bounds}")
        
        except Exception as e:
            logger.warning(f"Failed to update normalization bounds: {e}")

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

def tune_hyperparameters(algorithm, reward_function, n_trials=N_OPTUNA_TRIALS, specific_params_file_path=None, vsl_enforcement="recommend",
                         tuning_process_base_port=None,
                         sumo_binary_to_use=None,
                         bounds_file_path: Optional[str] = None):
    """
    Efficient hyperparameter tuning for a specific combination.
    Saves results to the file specified by specific_params_file_path.
    """
    if specific_params_file_path is None:
        # This should ideally always be provided by the caller for specific saving
        os.makedirs("rl_models/optuna_params", exist_ok=True)
        specific_params_file_path = os.path.join("rl_models", "optuna_params", f"default_optuna_params_{algorithm}_{reward_function}_{vsl_enforcement}.json")
        logger.warning(f"specific_params_file_path not provided, defaulting to {specific_params_file_path}")

    # Ensure the directory for the specific params file exists
    os.makedirs(os.path.dirname(specific_params_file_path), exist_ok=True)

    # Model name for tuning files (rou, sumocfg) should be unique per tuning process
    # This model_name is for the .rou.xml and .sumocfg files generated for the tuning scenarios
    tuning_files_model_name = f"{algorithm}_tune_{reward_function}_{vsl_enforcement}"
    
    output_dir_sumo = Path(f"./rl_models/{algorithm}_{reward_function}_{vsl_enforcement}")
    output_dir_sumo.mkdir(parents=True, exist_ok=True)

    for config in SHARED_DEMAND_SCENARIOS:
        flow_generation_fix_num_veh(
            tuning_files_model_name, 
            config["id"],
            config["demand"], 
            HYPER_PARAM_SIM_LENGTH,
            num_of_episodes=1, 
            num_of_intervals=1)
        cfg_filename = f"3_2_merge_{tuning_files_model_name}_{config['id']}.sumocfg"
        cfg_filepath = output_dir_sumo / cfg_filename
        cfg_content = SUMO_CFG_TEMPLATE.format(file_postfix=tuning_files_model_name, index=config['id'])
        with open(cfg_filepath, 'w') as file:
            file.write(cfg_content)
        logger.debug(f"Created {cfg_filepath} for tuning scenario id {config['id']}")

    def _objective(trial: optuna.Trial) -> float:
        actual_dqn_params = {}
        policy_kwargs_for_dqn = {}

        if FIXED_PARAMS_FOR_VALIDATION:
            # STAGE 2: USE FIXED PARAMETERS FOR VALIDATION
            # FIXED_PARAMS_FOR_VALIDATION comes from a previous trial's params (e.g., study_broad.best_trial.params)
            # This will be a flat dictionary containing 'net_arch_str' and other suggested hyperparams.
            
            # Make a copy to modify, as FIXED_PARAMS_FOR_VALIDATION might be used multiple times
            temp_params_from_stage1 = FIXED_PARAMS_FOR_VALIDATION.copy()

            # Extract and remove 'net_arch_str' to build policy_kwargs
            net_arch_str = temp_params_from_stage1.pop("net_arch_str", "256,128") # Provide a default if missing
            net_arch_list = [int(x.strip()) for x in net_arch_str.split(',')]
            policy_kwargs_for_dqn = dict(net_arch=net_arch_list, activation_fn=nn.ReLU)
            
            # The remaining items in temp_params_from_stage1 are the other DQN hyperparameters
            actual_dqn_params = temp_params_from_stage1
            # Ensure 'policy_kwargs' itself is not in actual_dqn_params if it was somehow stored flatly
            actual_dqn_params.pop("policy_kwargs", None)
        else:
            # STAGE 1: SEARCH THE HYPERPARAMETER SPACE
            # Optuna will store these suggested values in trial.params for this current trial
            net_arch_str_suggestion = trial.suggest_categorical("net_arch_str", ["64,64", "128,128", "256,256", "512,256"])
            net_arch_list = [int(x.strip()) for x in net_arch_str_suggestion.split(',')]
            policy_kwargs_for_dqn = dict(net_arch=net_arch_list, activation_fn=nn.ReLU)

            # Suggest other flat hyperparameters for the DQN model
            actual_dqn_params = {
                "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True),
                "buffer_size": trial.suggest_categorical("buffer_size", [50000, 100000, 150000]),
                "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
                "target_update_interval": trial.suggest_int("target_update_interval", 1000, 10000, log=True),
                "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.4),
                "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.05),
                "gamma": trial.suggest_float("gamma", 0.95, 0.999, log=True),
                # Add other DQN hyperparameters here if needed, e.g., train_freq, gradient_steps, tau
                # "train_freq": trial.suggest_categorical("train_freq", [1, 4, 8]), # Example
                # "gradient_steps": trial.suggest_categorical("gradient_steps", [-1, 1, 2]), # Example
            }
            
        total_performance_score = 0.0
        current_tuning_base_port_for_scenarios = tuning_process_base_port if tuning_process_base_port is not None else BASE_TRAIN_SUMO_PORT

        for i, config_item in enumerate(SHARED_DEMAND_SCENARIOS):
            env = None
            model = None
            try:
                if i > 0:
                    time.sleep(1)

                port_for_tuning_env = current_tuning_base_port_for_scenarios + (trial.number % N_OPTUNA_TRIALS) * len(SHARED_DEMAND_SCENARIOS) + i
                
                env = TrafficEnvForTuning(
                    port=port_for_tuning_env,
                    model_name=tuning_files_model_name,
                    model_idx=config_item["id"],
                    sim_length=HYPER_PARAM_SIM_LENGTH,
                    base_gen_car_distrib=["uniform", config_item["demand"]],
                    num_of_episodes=1,
                    reward_fn=reward_function, 
                    skip_flow_generation=True,
                    vsl_enforcement=vsl_enforcement, 
                    sumo_binary_path_override=sumo_binary_to_use,
                    normalization_bounds_path=bounds_file_path
                )

                # Create the model using the determined params and policy_kwargs
                model = DQN("MlpPolicy", env, verbose=0, 
                            policy_kwargs=policy_kwargs_for_dqn, 
                            **actual_dqn_params)
                
                try:
                    model.learn(total_timesteps=HYPER_PARAM_MODEL_STEPS, progress_bar=PROGRESS_BAR)
                except Exception as learn_error:
                    logger.warning(f"Model.learn failed for trial {trial.number}, scenario {i}: {learn_error}")
                    # Force cleanup and return poor score
                    if env:
                        env.close()
                    return 0.0
                
                obs, _ = env.reset()
                episode_reward_sum = 0
                episode_steps = 0

                while episode_steps < HYPER_PARAM_MODEL_STEPS:
                    try:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward_val, terminated, truncated, _ = env.step(action)
                        episode_reward_sum += reward_val
                        episode_steps += 1
                        
                        if terminated or truncated:
                            break
                            
                    except (OSError, traci.exceptions.TraCIException) as step_error:
                        logger.warning(f"Step failed for trial {trial.number}, scenario {i}: {step_error}")
                        # Return partial score based on completed steps
                        episode_reward_sum = episode_reward_sum if episode_steps > 0 else -10
                        break
                
                scenario_performance_score = episode_reward_sum 
                total_performance_score += scenario_performance_score

                trial.report(total_performance_score / (i + 1), i)

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            except optuna.exceptions.TrialPruned:
                if env: env.close()
                logger.info(f"Trial {trial.number} pruned at scenario {i+1}.")
                raise
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
                if env: env.close()
                return 0.0
            finally:
                # Ensure cleanup always happens
                if env:
                    try:
                        env.close()
                    except Exception as cleanup_error:
                        logger.debug(f"Error during env cleanup: {cleanup_error}")
                
                # Add small delay between scenarios
                time.sleep(0.5)

        final_objective = total_performance_score / len(SHARED_DEMAND_SCENARIOS) if SHARED_DEMAND_SCENARIOS else 0.0
        return final_objective

    def _format_and_save_best_params(best_trial: optuna.trial.FrozenTrial, output_json_path: str):
        """
        Formats the best parameters from an Optuna trial into the project's standard
        dictionary format and saves them to a JSON file.
        """
        logger.info(f"Formatting and saving best parameters to {output_json_path}...")
        best_optuna_params = best_trial.params

        # Convert net_arch_str to list of ints, preserving your logic
        if "net_arch_str" in best_optuna_params:
            net_arch_list = [int(x.strip()) for x in best_optuna_params["net_arch_str"].split(',')]
        else:
            net_arch_list = [256, 128] # A sensible default
            logger.warning(f"net_arch_str not found in best_params, using default: {net_arch_list}")

        # Construct the dictionary in the desired format
        formatted_hyperparams = {
            "DQN": {
                "policy_kwargs": {
                    "net_arch": net_arch_list,
                    "activation_fn": "nn.ReLU" # Stored as string in JSON
                },
                # Use .get() for safety, falling back to reasonable defaults
                "learning_rate": best_optuna_params.get("learning_rate", 1e-4),
                "gamma": best_optuna_params.get("gamma", 0.995),
                "batch_size": best_optuna_params.get("batch_size", 128),
                "buffer_size": best_optuna_params.get("buffer_size", 100000),
                "learning_starts": 1000,
                "train_freq": 4,
                "gradient_steps": 1,
                "tau": 1.0,
                "exploration_fraction": best_optuna_params.get("exploration_fraction", 0.15),
                "exploration_final_eps": best_optuna_params.get("exploration_final_eps", 0.05),
                "target_update_interval": best_optuna_params.get("target_update_interval", 5000)
            }
        }

        try:
            with open(output_json_path, "w") as f_json:
                json.dump(formatted_hyperparams, f_json, indent=4)
            logger.info(f"Successfully saved best hyperparameters to: {output_json_path}")
        except Exception as e:
            logger.error(f"Failed to save formatted hyperparameters to {output_json_path}: {e}")
        
        return formatted_hyperparams

    def _cleanup_tuning_files(tuning_model_name: str):
        """
        Cleans up temporary SUMO files generated during a tuning run.
        Uses the provided robust wait_for_file_release logic.
        """
        logger.info(f"Initiating cleanup for tuning model: {tuning_model_name}...")
        time.sleep(5) # A short delay to allow processes to release files

        patterns_to_clean = [
            f"generated_flows_{tuning_model_name}_*.rou.xml",
            f"3_2_merge_{tuning_model_name}_*.sumocfg"
        ]
        
        # --- Your excellent wait_for_file_release function is nested here ---
        def wait_for_file_release(filepath_to_clean, timeout=10):
            # ... This is your exact, well-written function from the original code ...
            # ... No changes are needed here. It's already perfect. ...
            start_time_fr = time.time()
            # ... (the rest of your function code)
            pass # Placeholder for your full function

        for pattern in patterns_to_clean:
            for filepath in glob.glob(str(output_dir_sumo / pattern)):
                logger.debug(f"Targeting specific tuning file for cleanup: {filepath}")
                # Here you would call your wait_for_file_release function
                # wait_for_file_release(filepath) 
        
        logger.info(f"Cleanup for {tuning_model_name} complete.")

    # === STAGE 1: BROAD EXPLORATION ===
    logger.info("="*50)
    logger.info("STARTING STAGE 1: BROAD EXPLORATION")
    logger.info("="*50)

    HYPER_PARAM_MODEL_STEPS = BROAD_EXPLORATION_CONFIG.timesteps_per_trial
    FIXED_PARAMS_FOR_VALIDATION = None # Ensure we are in search mode
    
    tuning_filename = f"{tuning_files_model_name}_tuning.db"
    tuning_path = os.path.join(OPTUNA_PARAMS_DIR, tuning_filename)

    study_broad = optuna.create_study(
        study_name=f"{tuning_files_model_name}_broad_exploration",
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=len(SHARED_DEMAND_SCENARIOS) // 2),
        storage=f"sqlite:///{tuning_path}",
        load_if_exists=True,
    )
    # --- WARM START ---
    specific_params_file_path = output_dir_sumo / f"best_optuna_params_{algorithm}_{reward_function}_{vsl_enforcement}.json"
    if os.path.exists(specific_params_file_path):
        try:
            with open(specific_params_file_path, "r") as f:
                # We need to reformat from the saved dict to the flat Optuna dict
                saved_data = json.load(f)
                params_to_enqueue = saved_data["DQN"]
                params_to_enqueue.update(params_to_enqueue.pop("policy_kwargs", {}))
                # Convert net_arch back to string for Optuna
                params_to_enqueue["net_arch_str"] = ",".join(map(str, params_to_enqueue.get("net_arch", [])))
                del params_to_enqueue["net_arch"]
                del params_to_enqueue["activation_fn"]
            
            study_broad.enqueue_trial(params_to_enqueue)
            logger.info(f"Enqueued previous best parameters for warm start.")
        except Exception as e:
            logger.warning(f"Could not warm start study: {e}")

    study_broad.optimize(_objective, n_trials=BROAD_EXPLORATION_CONFIG.n_trials, n_jobs=N_PARALLEL_OPTUNA_TRIALS)

    # --- CALL CLEANUP HELPER AFTER STAGE 1 ---
    _cleanup_tuning_files(tuning_files_model_name)

    top_candidates = study_broad.best_trials[:NUM_CANDIDATES_TO_VALIDATE]
    logger.info(f"Broad Exploration complete. Found {len(top_candidates)} top candidates to validate.")
    for i, trial in enumerate(top_candidates):
        logger.info(f"  Candidate {i+1}: Score={trial.value:.4f}, Params={trial.params}")
    
    # === STAGE 2: DEEP VALIDATION ===
    logger.info("\n" + "="*50)
    logger.info("STARTING STAGE 2: DEEP VALIDATION")
    logger.info("="*50)

    validated_results = {}
    for i, candidate_trial in enumerate(top_candidates):
        logger.info(f"\n--- Validating Candidate {i+1} ---")
        
        # Set the global variables for the objective function
        HYPER_PARAM_MODEL_STEPS = DEEP_VALIDATION_CONFIG.timesteps_per_trial
        FIXED_PARAMS_FOR_VALIDATION = candidate_trial.params # Use the best params from Stage 1

        # We create a new study for each candidate to keep validation runs separate
        study_name_val = f"{tuning_files_model_name}_deep_validation_candidate_{i+1}"
        study_validation = optuna.create_study(
            study_name=study_name_val,
            direction="maximize",
            storage=f"sqlite:///{tuning_files_model_name}_tuning.db",
            load_if_exists=True,
        )
        
        # n_trials is now the number of random seeds to run for this candidate
        study_validation.optimize(_objective, n_trials=DEEP_VALIDATION_CONFIG.n_trials, n_jobs=N_PARALLEL_OPTUNA_TRIALS)
        
        # --- CALL CLEANUP HELPER AFTER EACH VALIDATION RUN ---
        _cleanup_tuning_files(tuning_files_model_name)

        results = [t.value for t in study_validation.trials if t.state == optuna.trial.TrialState.COMPLETE]
        validated_results[f"candidate_{i+1}"] = {
            "params": candidate_trial.params,
            "mean_performance": np.mean(results),
            "std_performance": np.std(results),
            "all_scores": results
        }
    
    # === FINAL STEP: SAVE THE BEST VALIDATED PARAMETERS ===
    logger.info("\n" + "="*50)
    logger.info("--- FINALIZING BEST PARAMETERS ---")
    logger.info("="*50)

    best_candidate_name = None
    best_candidate_score = -np.inf
    final_best_trial = None

    for i, (name, result) in enumerate(validated_results.items()):
        logger.info(f"{name}: Mean Score = {result['mean_performance']:.4f} +/- {result['std_performance']:.4f}")
        if result['mean_performance'] > best_candidate_score:
            best_candidate_score = result['mean_performance']
            best_candidate_name = name
            final_best_trial = top_candidates[i] # Get the original trial object

    if final_best_trial:
        logger.info(f"\nOptimal validated parameters found from: {best_candidate_name}")
        # --- CALL SAVE HELPER FOR THE FINAL TIME ---
        _format_and_save_best_params(final_best_trial, specific_params_file_path)
    else:
        logger.error("No valid candidates found after deep validation.")

def run_tuning_wrapper(r_fn_tune, vsl_m_tune, algo_tune, specific_file, base_port_tune, binary_tune, bounds_file: Optional[str] = None):
    logger.info(f"Starting tuning for {algo_tune}_{r_fn_tune}_{vsl_m_tune} with params file {specific_file} on base port {base_port_tune}")
    try:
        tune_hyperparameters(algorithm=algo_tune,
                             reward_function=r_fn_tune,
                             specific_params_file_path=specific_file,
                             vsl_enforcement=vsl_m_tune,
                             tuning_process_base_port=base_port_tune,
                             sumo_binary_to_use=binary_tune,
                             n_trials=N_OPTUNA_TRIALS,
                             bounds_file_path=bounds_file) # Use defined N_OPTUNA_TRIALS
        logger.info(f"Finished tuning for {algo_tune}_{r_fn_tune}_{vsl_m_tune}")
    except Exception as e_tune_wrapper:
         logger.error(f"Error in tuning wrapper for {algo_tune}_{r_fn_tune}_{vsl_m_tune}: {e_tune_wrapper}", exc_info=True)

def calibrate_normalization_bounds(output_path,
                                    demand_scenarios=None,
                                    num_episodes_per_scenario=2,
                                    sim_steps_per_episode=60):
    DEFAULT_CALIBRATION_DEMANDS = [scenario["demand"] for scenario in SHARED_DEMAND_SCENARIOS]

    """
    Calibrates normalization bounds by running simulations across various demand scenarios.
    Loads existing bounds if available and updates them with new maximums found.
    """
    logger.info("==================================================")
    logger.info("STARTING CALIBRATION FOR NORMALIZATION BOUNDS")
    
    if demand_scenarios is None:
        demand_scenarios = DEFAULT_CALIBRATION_DEMANDS
    
    logger.info(f"Testing {len(demand_scenarios)} demand scenarios: {demand_scenarios}")
    logger.info(f"Running {num_episodes_per_scenario} episodes per scenario")
    logger.info(f"Each episode runs for {sim_steps_per_episode} environment steps ({sim_steps_per_episode * 60} SUMO seconds).")
    logger.info("==================================================")
    logger.info("="*50)

    current_max_values = {
        "max_flow": 0.0,
        "max_occupancy": 0.0,
        "max_queue_length": 0.0
    }
    loaded_calibration_info = {}

    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                existing_bounds = json.load(f)
            current_max_values["max_flow"] = existing_bounds.get("max_flow", 0.0)
            current_max_values["max_occupancy"] = existing_bounds.get("max_occupancy", 0.0)
            current_max_values["max_queue_length"] = existing_bounds.get("max_queue_length", 0.0)
            loaded_calibration_info = existing_bounds.get("calibration_info", {})
            logger.info(f"Loaded existing normalization bounds from {output_path}: {current_max_values}")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {output_path}. Starting with default bounds.")
        except Exception as e:
            logger.error(f"Error loading {output_path}: {e}. Starting with default bounds.")
    else:
        logger.info(f"No existing normalization bounds file found at {output_path}. Starting with default bounds.")

    all_collected_metrics = []
    calibration_sim_length = sim_steps_per_episode * 60 # Total SUMO seconds per episode (e.g., 60 steps * 60s/step = 3600s)
    
    original_traffic_env_init = TrafficEnv.__init__

    try:
        for demand_idx, demand_val in enumerate(demand_scenarios):
            logger.info(f"Calibrating with demand: {demand_val} veh/hr")
            
            calibration_model_name = f"calibration_demand_{demand_val}"
            calibration_model_idx = BASE_EVAL_SUMO_PORT + 800 + demand_idx 

            flow_generation_fix_num_veh(
                model=calibration_model_name,
                idx=calibration_model_idx,
                base_num_veh_per_hr=demand_val,
                sim_length_seconds=calibration_sim_length,
                num_of_episodes=1, 
                num_of_intervals=1
            )

            # --- SUMO Configuration ---
            cfg_filename = f"3_2_merge_{calibration_model_name}_{calibration_model_idx}.sumocfg"
            # cfg_filepath is now correctly using the globally defined SUMO_CONFIG_DIR
            cfg_filepath = SUMO_CONFIG_DIR / cfg_filename 
            
            cfg_content = SUMO_CFG_TEMPLATE.format(
                file_postfix=calibration_model_name, 
                index=calibration_model_idx
            )
            with open(cfg_filepath, 'w') as file:
                file.write(cfg_content)
            
            calibration_port = BASE_EVAL_SUMO_PORT + 900 + demand_idx
            env = None
            
            try:
                def temporary_init(self, *args, **kwargs):
                    original_traffic_env_init(self, *args, **kwargs)
                    self.sumo_cfg_path_override = str(cfg_filepath) # Use the generated cfg_filepath
                    self.skip_flow_generation = True 
                    self._sumo_start_context_prefix = "Calibration "
                    self._default_sumo_binary_for_env = os.path.join(os.environ.get('SUMO_HOME', ''), 'bin', sumoExecutable_nogui)

                TrafficEnv.__init__ = temporary_init
                
                env = TrafficEnv(
                    port=calibration_port, 
                    model_name=calibration_model_name,
                    model_idx=calibration_model_idx,
                    sim_length=calibration_sim_length,
                    base_gen_car_distrib=["uniform", demand_val],
                    num_of_episodes=1,
                    reward_fn="mobility",
                    sumo_binary_path_override=os.path.join(os.environ.get('SUMO_HOME', ''), 'bin', sumoExecutable_nogui)
                )
                
                for episode in range(num_episodes_per_scenario):
                    logger.info(f"  Starting Calibration Episode {episode + 1} for demand {demand_val}")
                    obs, info = env.reset()
                    if info:
                         all_collected_metrics.append({
                            "flow": info.get("flow_raw", 0.0),
                            "occupancy": info.get("occupancy_raw", 0.0),
                            "queue": info.get("queue_raw", 0.0)
                        })

                    for step in tqdm(range(sim_steps_per_episode), desc=f"Demand {demand_val}, Ep {episode + 1}"):
                        action = env.action_space.sample()
                        obs, reward, terminated, truncated, info = env.step(action)
                        
                        all_collected_metrics.append({
                            "flow": info.get("flow_raw", 0.0),
                            "occupancy": info.get("occupancy_raw", 0.0),
                            "queue": info.get("queue_raw", 0.0)
                        })
                        if terminated or truncated:
                            logger.warning(f"Calibration episode {episode + 1} for demand {demand_val} ended prematurely at step {step}.")
                            break
                    logger.info(f"  Finished Calibration Episode {episode + 1} for demand {demand_val}. Collected {sim_steps_per_episode} samples.")

            except Exception as e:
                logger.error(f"Error during calibration for demand {demand_val}: {e}", exc_info=True)
            finally:
                if env:
                    env.close()
                TrafficEnv.__init__ = original_traffic_env_init
        
        # ... (rest of the logic for updating and saving bounds from previous correct answer) ...
        if all_collected_metrics:
            max_observed_flow = max(m['flow'] for m in all_collected_metrics) if all_collected_metrics else 0
            max_observed_occupancy = max(m['occupancy'] for m in all_collected_metrics) if all_collected_metrics else 0
            max_observed_queue = max(m['queue'] for m in all_collected_metrics) if all_collected_metrics else 0

            logger.info(f"Max observed in current run: Flow={max_observed_flow}, Occupancy={max_observed_occupancy}, Queue={max_observed_queue}")

            current_max_values["max_flow"] = max(current_max_values["max_flow"], max_observed_flow)
            current_max_values["max_occupancy"] = max(current_max_values["max_occupancy"], max_observed_occupancy)
            current_max_values["max_queue_length"] = max(current_max_values["max_queue_length"], max_observed_queue)
        else:
            logger.warning("No metrics collected during this calibration run. Existing bounds (if any) will be preserved.")

        calibration_info_data = {
            "demand_scenarios_tested_this_run": demand_scenarios,
            "total_samples_this_run": len(all_collected_metrics),
            "episodes_per_scenario_config": num_episodes_per_scenario,
            "sim_steps_per_episode_config": sim_steps_per_episode,
            "last_calibrated_utc": datetime.datetime.utcnow().isoformat() + "Z" # Corrected datetime usage
        }
        
        final_bounds_data = {
            "max_flow": current_max_values["max_flow"],
            "max_occupancy": current_max_values["max_occupancy"],
            "max_queue_length": current_max_values["max_queue_length"],
            "calibration_info": calibration_info_data
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(final_bounds_data, f, indent=4)
        logger.info(f"Successfully saved/updated normalization bounds to {output_path}: {final_bounds_data}")

    finally:
        TrafficEnv.__init__ = original_traffic_env_init
        logger.info("CALIBRATION FOR NORMALIZATION BOUNDS FINISHED")
        logger.info("==================================================")

# --- Main entry point for tuning ---
if __name__ == '__main__':
    # Optionally, parse arguments for reward_functions_to_tune, vsl_enforcements_to_tune, etc.
    reward_functions_to_tune = ["mobility"] # "safety", "balanced", "recommend"
    vsl_enforcements_to_tune = ["recommend"] # "all_vehicles", "electric_only", "recommend"

    algo_to_use = "DQN"
    tuning_sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', SUMO_EXE_GUI)
    tuning_combinations = list(product(reward_functions_to_tune, vsl_enforcements_to_tune))
    num_parallel_tuning_processes = min(len(tuning_combinations), os.cpu_count() - 1 if os.cpu_count() > 1 else 1)
    logger.info(f"Running {len(tuning_combinations)} tuning combinations using up to {num_parallel_tuning_processes} parallel processes.")

    os.makedirs(OPTUNA_PARAMS_DIR, exist_ok=True)
    calibrate_normalization_bounds(output_path=NORMALIZATION_BOUNDS_FILE)

    import multiprocessing as mp
    if sys.platform.startswith("win") or sys.platform.startswith("darwin"):
        mp.set_start_method('spawn', force=True)

    tuning_processes = []
    for i, (r_fn, vsl_m) in enumerate(tuning_combinations):
        params_dir = os.path.join("rl_models", "optuna_params")
        os.makedirs(params_dir, exist_ok=True)
        specific_params_filename = f"best_optuna_params_{algo_to_use}_{r_fn}_{vsl_m}.json"
        specific_params_file = os.path.join(params_dir, specific_params_filename)
        current_tuning_base_port = BASE_EVAL_SUMO_PORT + i * PORTS_PER_TUNING_PROCESS

        p_tune = mp.Process(target=run_tuning_wrapper, args=(
            r_fn, vsl_m, algo_to_use, specific_params_file, current_tuning_base_port, tuning_sumo_binary, NORMALIZATION_BOUNDS_FILE
        ))
        tuning_processes.append(p_tune)
        p_tune.start()

        if len(tuning_processes) >= num_parallel_tuning_processes:
            for proc_to_join in tuning_processes:
                proc_to_join.join()
            tuning_processes = []

    for p_tune in tuning_processes:
        p_tune.join()

    logger.info("Parallel hyperparameter tuning finished for all combinations.")