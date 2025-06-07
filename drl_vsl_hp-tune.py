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
import torch as th

# Import your TrafficEnv and any other shared code from drl_vsl.py
from drl_vsl import (
    TrafficEnv,
    SUMO_CFG_TEMPLATE,
    PROGRESS_BAR_ENABLED,
    BASE_TRAIN_SUMO_PORT,
    BASE_EVAL_SUMO_PORT,
    PORTS_PER_TUNING_PROCESS,
    HYPER_PARAM_SIM_LENGTH,
    HYPER_PARAM_OPTUNA_STUD_TIMEOUT,
    HYPER_PARAM_MODEL_STEPS,
    N_OPTUNA_TRIALS,
    sumoExecutable_nogui,
    flow_generation_fix_num_veh,
    logger,
)

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

# --- tune_hyperparameters function ---
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
    
    output_dir_sumo = Path("./traffic_environment/sumo")
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

    def objective(trial: optuna.Trial) -> float:
        net_arch_str_suggestion = trial.suggest_categorical("net_arch_str", ["64, 64", # Small
                                                                             "128,128", # Medium
                                                                             "256,256", # Large
                                                                             "512,256" # Asymmetrical, deeper network
                                                                             ])
        net_arch_list = [int(x) for x in net_arch_str_suggestion.split(',')]
        # activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
        # activation_fn = {"tanh": th.nn.Tanh, "relu": th.nn.ReLU}[activation_fn_name]
        policy_kwargs = dict(net_arch=net_arch_list, activation_fn=nn.ReLU) # Use nn.ReLU directly isntead of activation_fn
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True), # Rates > 1e-3 are often unstable with Adam.
            "buffer_size": trial.suggest_categorical("buffer_size", [50000, 100000, 150000]), # A buffer of 50k-100k is often sufficient for learning key dynamics in shorter runs.
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]), # Larger batches provide more stable gradients.
            "target_update_interval": trial.suggest_int("target_update_interval", 1000, 10000), # A wider log search is good
            "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.4), # For short tuning runs, exploration needs to be significant.
            "exploration_initial_eps": 1.0, ## Start with full exploration
            "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.05),
            "learning_starts": 1000, # A fixed, reasonable value
            "train_freq": 4, # A common and effective value
            "gradient_steps": 1, # Corresponds to train_freq=4
            "tau": 1, # Hard updates are standard
            "gamma": trial.suggest_float("gamma", 0.95, 0.999), # A log scale to focus the search on values close to 1.0 is used
        }
        total_performance_score = 0.0
        current_tuning_base_port_for_scenarios = tuning_process_base_port if tuning_process_base_port is not None else BASE_TRAIN_SUMO_PORT

        for i, config_item in enumerate(scenario_configs):
            env = None
            try:
                port_for_tuning_env = current_tuning_base_port_for_scenarios + (trial.number % N_OPTUNA_TRIALS) * len(scenario_configs) + i
                
                # Use the specialized environment for tuning
                env = TrafficEnvForTuning(
                    port=port_for_tuning_env,
                    model_name=tuning_files_model_name,
                    model_idx=config_item["id"],
                    op_mode="train",
                    base_gen_car_distrib=["uniform", config_item["demand"]],
                    reward_fn=reward_function, skip_flow_generation=True,
                    vsl_enforcement=vsl_enforcement, sumo_binary_path_override=sumo_binary_to_use
                )

                model = DQN("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs, **params)
                
                # Learn on the environment
                model.learn(total_timesteps=HYPER_PARAM_MODEL_STEPS, progress_bar=False) # Progress bar off for cleaner logs
                
                # --- 2. ENHANCED EVALUATION & OBJECTIVE ---
                # Now, evaluate the learned policy to get the final metrics
                obs, _ = env.reset()
                while True:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = env.step(action)
                    if terminated or truncated:
                        break
                
                # Get the comprehensive summary statistics from your logger
                summary_stats = env.logger.get_summary_statistics()
                
                # The objective is now the composite performance score!
                # This directly optimizes for the balance of safety and mobility.
                scenario_performance_score = summary_stats.get('performance_score', 0.0)
                total_performance_score += scenario_performance_score

                # --- 3. INTEGRATED PRUNING ---
                # Report the intermediate performance to Optuna
                trial.report(total_performance_score / (i + 1), i)

                # Check if the trial should be pruned
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            except optuna.exceptions.TrialPruned:
                # Propagate the pruning exception
                if env: env.close()
                raise
            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
                if env: env.close()
                return 0.0 # Return a poor score for failed trials
            finally:
                if env: env.close()

        # The final return value is the average performance score across all scenarios
        final_objective = total_performance_score / len(scenario_configs)
        return final_objective

    study = optuna.create_study(direction='maximize')
    
    # Try to load from the specific params file for warm start
    if os.path.exists(specific_params_file_path):
        try:
            with open(specific_params_file_path, "r") as f:
                prev_best_params = json.load(f)
            study.enqueue_trial(prev_best_params)
            logger.info(f"Enqueued previous best parameters from {specific_params_file_path} for warm start of {tuning_files_model_name}.")
        except Exception as e:
            logger.info(f"Could not load or enqueue previous Optuna params from {specific_params_file_path} for {tuning_files_model_name}: {e}")
    else:
        logger.info(f"Specific Optuna params file {specific_params_file_path} not found for {tuning_files_model_name}. Starting fresh study.")

    study.optimize(objective, n_trials=n_trials, timeout=HYPER_PARAM_OPTUNA_STUD_TIMEOUT)

    logger.info(f"Optuna study for {tuning_files_model_name} (params for {algorithm}_{reward_function}_{vsl_enforcement}) completed. Best params: {study.best_params}")

    best_optuna_params = study.best_params
    # Convert net_arch_str to list of ints if it exists
    if "net_arch_str" in best_optuna_params:
        net_arch_list = [int(x.strip()) for x in best_optuna_params["net_arch_str"].split(',')]
    else:
        # Fallback if net_arch_str was not tuned or not found in best_params
        net_arch_list = [512, 256, 128] # Default or from your ENHANCED_HYPERPARAMS
        logger.warning(f"net_arch_str not found in Optuna best_params, using default: {net_arch_list}")

    # Construct the dictionary in the desired format
    formatted_hyperparams = {
        "DQN": {
            "policy_kwargs": {
                "net_arch": net_arch_list,
                "activation_fn": "nn.ReLU"  # Placeholder, will be replaced with actual object
            },
            "learning_rate": best_optuna_params.get("learning_rate", 1e-4),
            "gamma": best_optuna_params.get("gamma", 0.995),
            "batch_size": best_optuna_params.get("batch_size", 64),
            "train_freq": (best_optuna_params.get("train_freq", 4), "step"), # Ensure tuple format
            "gradient_steps": best_optuna_params.get("gradient_steps", 1),
            "tau": best_optuna_params.get("tau", 1.0),
            "buffer_size": best_optuna_params.get("buffer_size", 250000),
            "learning_starts": best_optuna_params.get("learning_starts", 10000),
            "exploration_fraction": best_optuna_params.get("exploration_fraction", 0.20),
            "exploration_initial_eps": best_optuna_params.get("exploration_initial_eps", 1.0),
            "exploration_final_eps": best_optuna_params.get("exploration_final_eps", 0.01),
            "target_update_interval": best_optuna_params.get("target_update_interval", 10000)
        }
    }

    py_file_path = specific_params_file_path.replace(".json", ".py")

    try:
        with open(specific_params_file_path, "w") as f_json:
            # Convert activation_fn to string for JSON compatibility
            formatted_hyperparams_json = formatted_hyperparams.copy()
            formatted_hyperparams_json["DQN"]["policy_kwargs"]["activation_fn"] = "nn.ReLU"
            json.dump(formatted_hyperparams_json, f_json, indent=4)
        logger.info(f"Saved best Optuna params in ENHANCED_HYPERPARAMS JSON format to: {specific_params_file_path}")
    except Exception as e_save_py:
        logger.error(f"Failed to save formatted hyperparameters to {py_file_path}: {e_save_py}")
    
    delay_before_cleanup = 10 
    logger.info(f"Waiting {delay_before_cleanup} seconds before cleaning up tuning files for {tuning_files_model_name}...")
    time.sleep(delay_before_cleanup)

    patterns_to_clean = [
        f"generated_flows_{tuning_files_model_name}_*.rou.xml", # Use tuning_files_model_name
        f"3_2_merge_{tuning_files_model_name}_*.sumocfg"    # Use tuning_files_model_name
    ]

    # ... (wait_for_file_release function remains the same) ...
    def wait_for_file_release(filepath_to_clean, timeout=10): # Increased default timeout
        start_time_fr = time.time()
        file_path_obj_fr = Path(filepath_to_clean)

        if not file_path_obj_fr.exists():
            logger.debug(f"File {filepath_to_clean} does not exist. No need to remove.")
            return True

        logger.debug(f"Attempting to remove {filepath_to_clean}...")
        while time.time() - start_time_fr < timeout:
            try:
                os.remove(filepath_to_clean)
                logger.debug(f"Successfully removed: {filepath_to_clean}")
                return True
            except FileNotFoundError: # If removed by another process or in a previous attempt
                logger.debug(f"File {filepath_to_clean} already gone (FileNotFoundError during retry).")
                return True
            except PermissionError as e_perm_fr: # Specifically catch PermissionError (WinError 32)
                logger.warning(f"Could not remove {filepath_to_clean} due to PermissionError (likely in use): {e_perm_fr}. Retrying in 1s...")
                time.sleep(1)
            except Exception as e_fr: # Catch other potential OS errors
                logger.warning(f"Could not remove {filepath_to_clean} due to OS error: {e_fr}. Retrying in 1s...")
                time.sleep(1)
        
        logger.error(f"Failed to remove {filepath_to_clean} after {timeout} seconds. It might still be in use.")
        if file_path_obj_fr.exists(): # Check one last time
            logger.error(f"File {filepath_to_clean} STILL EXISTS. Listing active SUMO processes:")
            try:
                for proc in psutil.process_iter(['pid', 'name']): # Removed 'username' for brevity/permission
                    if 'sumo' in proc.info['name'].lower():
                        logger.error(f"  Potential SUMO culprit: PID {proc.info['pid']}, Name {proc.info['name']}")
            except (psutil.Error) as e_psutil: # Catch all psutil errors
                 logger.error(f"Could not list processes due to psutil error: {e_psutil}")
        return False
        
    for pattern in patterns_to_clean:
        # Glob directly in the sumo directory
        for filepath_to_clean_glob in glob.glob(str(output_dir_sumo / pattern)):
            # The pattern already includes tuning_files_model_name, so it's specific enough
            logger.debug(f"Targeting specific tuning file for cleanup: {filepath_to_clean_glob}")
            wait_for_file_release(filepath_to_clean_glob)
    
    return study.best_params

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
    reward_functions_to_tune = ["mobility", "safety"]
    vsl_enforcements_to_tune = ["all_vehicles", "electric_only", "recommend"]

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