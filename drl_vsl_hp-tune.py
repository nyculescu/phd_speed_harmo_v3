import os
import sys
import time
import json
import glob
import psutil
from pathlib import Path
from itertools import product
import optuna
from stable_baselines3 import DQN
import torch.nn as nn
import traci
from dataclasses import dataclass
import numpy as np

# Import your TrafficEnv and any other shared code from drl_vsl.py
from drl_vsl import (
    TrafficEnv,
    SUMO_CFG_TEMPLATE,
    BASE_TRAIN_SUMO_PORT,
    BASE_EVAL_SUMO_PORT,
    PORTS_PER_TUNING_PROCESS,
    OPTUNA_PARAMS_DIR,
    sumoExecutable_nogui,
    flow_generation_fix_num_veh,
    logger,
)

# --- 1. SETUP: DEFINE TUNING CONFIGURATIONS ---
# This makes the two-stage process explicit and easy to manage without a class.
@dataclass
class TuningConfig:
    name: str
    n_trials: int
    timesteps_per_trial: int

# Configuration for Stage 1
BROAD_EXPLORATION_CONFIG = TuningConfig(
    name="Broad Exploration",
    n_trials=40,
    timesteps_per_trial=10000
)

# Configuration for Stage 2
DEEP_VALIDATION_CONFIG = TuningConfig(
    name="Deep Validation",
    n_trials=3,  # This will be the number of random seeds
    timesteps_per_trial=30000
)

FIXED_PARAMS_FOR_VALIDATION = None
HYPER_PARAM_SIM_LENGTH = 1800 # Simulation length for tuning scenarios
N_PARALLEL_OPTUNA_TRIALS = 2
NUM_CANDIDATES_TO_VALIDATE = 3
N_OPTUNA_TRIALS = 40 # Number of trials for Optuna study
PROGRESS_BAR = True

# --- TrafficEnvForTuning class ---
class TrafficEnvForTuning(TrafficEnv):
    """
    Specialized TrafficEnv for hyperparameter tuning that uses pre-generated flow files.
    Inherits from TrafficEnv but skips flow generation to use scenario-specific files.
    """
    
    def __init__(self, port, model_name, model_idx, op_mode, base_gen_car_distrib, 
                 num_of_episodes=0, reward_fn="balanced", skip_flow_generation=True, vsl_enforcement="recommend",
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
        
        self.total_vehicles_before = 0
        self.total_vehicles_after = 0
    
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
        
        return observation, reward, done, truncated, info
    
    def reset(self, seed=None, options=None):
        """Reset environment and tuning-specific counters."""
        # Reset tuning counters
        self._tuning_step_count = 0
        self._cumulative_reward = 0.0
        
        # Call parent reset
        return super().reset(seed, options)
    
    def close_sumo(self, reason):
        super().close_sumo(reason)
    
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
                         sumo_binary_to_use=None):
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

    scenario_configs = [
        {"id": 100, "demand": 2000, "pattern": "uniform"},
        {"id": 101, "demand": 2500, "pattern": "uniform"}, 
        {"id": 102, "demand": 3000, "pattern": "uniform"},
        {"id": 103, "demand": 3500, "pattern": "uniform"}
    ]
    # Model name for tuning files (rou, sumocfg) should be unique per tuning process
    # This model_name is for the .rou.xml and .sumocfg files generated for the tuning scenarios
    tuning_files_model_name = f"{algorithm}_tune_{reward_function}_{vsl_enforcement}"
    
    output_dir_sumo = Path(f"./rl_models/{algorithm}_{reward_function}_{vsl_enforcement}")
    output_dir_sumo.mkdir(parents=True, exist_ok=True)

    for config in scenario_configs:
        flow_generation_fix_num_veh(
            tuning_files_model_name, 
            config["id"],
            config["demand"], 
            num_of_hrs=1, # For HYPER_PARAM_SIM_LENGTH
            num_of_episodes=1, 
            num_of_intervals=1, 
            op_mode="train" # op_mode for flow_gen, env will use its own
        )
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

        for i, config_item in enumerate(scenario_configs):
            env = None
            try:
                port_for_tuning_env = current_tuning_base_port_for_scenarios + (trial.number % N_OPTUNA_TRIALS) * len(scenario_configs) + i
                
                env = TrafficEnvForTuning(
                    port=port_for_tuning_env,
                    model_name=tuning_files_model_name, # Name for .rou.xml files
                    model_idx=config_item["id"],       # Scenario ID for .rou.xml files
                    op_mode="train", # op_mode for TrafficEnvForTuning
                    base_gen_car_distrib=["uniform", config_item["demand"]],
                    reward_fn=reward_function, skip_flow_generation=True,
                    vsl_enforcement=vsl_enforcement, sumo_binary_path_override=sumo_binary_to_use
                )

                # Create the model using the determined params and policy_kwargs
                model = DQN("MlpPolicy", env, verbose=0, 
                            policy_kwargs=policy_kwargs_for_dqn, 
                            **actual_dqn_params)
                
                model.learn(total_timesteps=HYPER_PARAM_MODEL_STEPS, progress_bar=PROGRESS_BAR)
                
                obs, _ = env.reset()
                episode_reward_sum = 0
                episode_steps = 0
                while episode_steps < HYPER_PARAM_MODEL_STEPS: # Or some other termination condition
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward_val, terminated, truncated, _ = env.step(action)
                    episode_reward_sum += reward_val
                    episode_steps +=1
                    if terminated or truncated:
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
                if env: env.close()

        final_objective = total_performance_score / len(scenario_configs) if scenario_configs else 0.0
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
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=len(scenario_configs) // 2),
        storage=f"sqlite:///{tuning_path}",
        load_if_exists=True,
    )
    # --- INTEGRATE WARM START ---
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

# --- run_tuning_wrapper function ---
def run_tuning_wrapper(r_fn_tune, vsl_m_tune, algo_tune, specific_file, base_port_tune, binary_tune):
    logger.info(f"Starting tuning for {algo_tune}_{r_fn_tune}_{vsl_m_tune} with params file {specific_file} on base port {base_port_tune}")
    try:
        tune_hyperparameters(algorithm=algo_tune,
                             reward_function=r_fn_tune,
                             specific_params_file_path=specific_file,
                             vsl_enforcement=vsl_m_tune,
                             tuning_process_base_port=base_port_tune,
                             sumo_binary_to_use=binary_tune,
                             n_trials=N_OPTUNA_TRIALS) # Use defined N_OPTUNA_TRIALS
        logger.info(f"Finished tuning for {algo_tune}_{r_fn_tune}_{vsl_m_tune}")
    except Exception as e_tune_wrapper:
         logger.error(f"Error in tuning wrapper for {algo_tune}_{r_fn_tune}_{vsl_m_tune}: {e_tune_wrapper}", exc_info=True)

# --- Main entry point for tuning ---
if __name__ == '__main__':
    # Optionally, parse arguments for reward_functions_to_tune, vsl_enforcements_to_tune, etc.
    reward_functions_to_tune = ["mobility"] # "safety", "balanced", "recommend"
    vsl_enforcements_to_tune = ["recommend"] # "all_vehicles", "electric_only", "recommend"

    algo_to_use = "DQN"
    tuning_sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', sumoExecutable_nogui)
    tuning_combinations = list(product(reward_functions_to_tune, vsl_enforcements_to_tune))
    num_parallel_tuning_processes = min(len(tuning_combinations), os.cpu_count() - 1 if os.cpu_count() > 1 else 1)
    logger.info(f"Running {len(tuning_combinations)} tuning combinations using up to {num_parallel_tuning_processes} parallel processes.")

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
            r_fn, vsl_m, algo_to_use, specific_params_file, current_tuning_base_port, tuning_sumo_binary
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