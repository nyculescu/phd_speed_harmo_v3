import logging
from stable_baselines3 import DQN, A2C, PPO
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from flow_gen import *
from gymnasium.wrappers import TimeLimit
import gymnasium as gym
from gymnasium import spaces
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
sumoExecutable = 'sumo-gui.exe' if os.name == 'nt' else 'sumo-gui'
sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', sumoExecutable)

MAX_OCCUPANCY = 100.0  # Occupancy percentage
MAX_FLOW = 7200.0      # vehicles/hour (theoretical maximum for 2 lanes)
MAX_SPEED_DIFF = 80.0  # km/h (130 - 50)
MAX_QUEUE_LENGTH = 500 # vehicles (adjust based on your segment length)
OBSERVATION_SPACE_SIZE = 7

# Enhanced hyperparameters based on traffic control research
ENHANCED_HYPERPARAMS = {
    "DQN": {
        "learning_rate": 5e-4,
        "buffer_size": 200000,
        "batch_size": 64,
        "target_update_interval": 5000,
        "exploration_fraction": 0.15,
        "exploration_final_eps": 0.02,
        "net_arch": [512, 256, 128]
    },
    "PPO": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gamma": 0.995,
        "net_arch": [256, 256]
    },
    "A2C": {
        "learning_rate": 7e-4,
        "gamma": 0.99,
        "net_arch": [256, 128]
    }
}

def create_sumocfg(model):
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
        # Create filename based on model and environment index
        filename = f"3_2_merge_{model}_{i}.sumocfg"
        filepath = os.path.join(output_dir, filename)
        
        # Format the template with current model and index
        content = sumocfg_template.format(model=model, index=i)
        
        # Write the content to the file
        with open(filepath, 'w') as file:
             file.write(content)
        
        logging.debug(f"Created {filepath}")

def train_env_constructor(idx, model_name, num_of_episodes, reward_fn): #Pass reward function
    def _init():
        env = Monitor(TrafficEnv(port=base_train_sumo_port + idx,
                                model_name=model_name,
                                model_idx=idx,
                                op_mode="train",
                                base_gen_car_distrib=["uniform", 2000],
                                num_of_episodes=num_of_episodes,
                                reward_fn=reward_fn)) #Pass reward function
        return env
    return _init

def eval_env_constructor(model_name, reward_fn): #Pass reward function
    def _init():
        env = Monitor(TimeLimit(TrafficEnv(port=base_eval_sumo_port,
                                            model_name=model_name,
                                            model_idx=num_envs_per_model - 1,
                                            op_mode="eval",
                                            base_gen_car_distrib=["uniform", 3000],
                                            num_of_episodes=1, #Fixed to 1
                                            reward_fn=reward_fn), #Pass reward function
                                max_episode_steps=interval_length))
        return env
    return _init

def train_model(algorithm,
                reward_function="balanced", 
                num_of_episodes=7,
                use_enhanced_params=True,
                custom_params=None):
    """
    Train RL model using research-optimized hyperparameters.
    
    Args:
        algorithm (str): RL algorithm ("DQN", "A2C", "PPO")
        reward_function (str): Reward function type
        num_of_episodes (int): Training episodes
        use_enhanced_params (bool): Use ENHANCED_HYPERPARAMS
        custom_params (dict): Override specific hyperparameters
    """

    if use_enhanced_params and algorithm in ENHANCED_HYPERPARAMS:
        params = ENHANCED_HYPERPARAMS[algorithm].copy()
        logging.info(f"Using enhanced hyperparameters for {algorithm}")
    else:
        # Fallback to default parameters
        params = {
            "learning_rate": 1e-4,
            "batch_size": 128,
            "gamma": 0.99,
            "net_arch": [256, 256, 128]
        }
        logging.warning(f"Using default hyperparameters for {algorithm}")

    if custom_params:
        params.update(custom_params)
        logging.info(f"Applied custom parameter overrides: {custom_params}")

    steps_per_episode = 504000 // 60
    total_timesteps = steps_per_episode * num_of_episodes
    eval_timesteps = steps_per_episode // 4
    
    # model_name = f"{algorithm}_{reward_function}"
    model_name = algorithm
    log_dir = f"./logs/{model_name}/"
    model_dir = f"./rl_models/{model_name}/"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Select reward function
    # reward_fn_map = {"mobility": "mobility", "safety": "safety", "balanced": "balanced"}
    # reward_fn = reward_fn_map.get(reward_function)
    # if not reward_fn:
    #     raise ValueError("Invalid reward function name")
    
    train_env = SubprocVecEnv([
        train_env_constructor(i, model_name, num_of_episodes, reward_function)
        for i in range(num_train_envs_per_model)
    ])
    env_eval = SubprocVecEnv([eval_env_constructor(model_name, reward_function)])
    
    # Model configuration
    policy_kwargs = dict(
        net_arch=params["net_arch"], 
        activation_fn=nn.ReLU
    )
    
    # Initialize algorithm[3]
    if algorithm == "DQN":
        model = DQN("MlpPolicy", train_env, 
                   learning_rate=params["learning_rate"],
                   buffer_size=params["buffer_size"],
                   batch_size=params["batch_size"],
                   gamma=params.get("gamma", 0.99),
                   target_update_interval=params["target_update_interval"],
                   exploration_fraction=params["exploration_fraction"],
                   exploration_final_eps=params["exploration_final_eps"],
                   policy_kwargs=policy_kwargs,
                   verbose=1, tensorboard_log=log_dir, device='cuda')
                   
    elif algorithm == "PPO":
        model = PPO("MlpPolicy", train_env,
                   learning_rate=params["learning_rate"],
                   n_steps=params["n_steps"],
                   batch_size=params["batch_size"],
                   gamma=params["gamma"],
                   policy_kwargs=policy_kwargs,
                   verbose=1, tensorboard_log=log_dir, device='cuda')
                   
    elif algorithm == "A2C":
        model = A2C("MlpPolicy", train_env,
                   learning_rate=params["learning_rate"],
                   gamma=params["gamma"],
                   policy_kwargs=policy_kwargs,
                   verbose=1, tensorboard_log=log_dir, device='cuda')
    
    # Configure callbacks for proper evaluation[3]
    checkpoint_cb = CheckpointCallback(
        save_freq=eval_timesteps,
        save_path=model_dir,
        name_prefix=f"rl_model_{model_name}",
        save_replay_buffer=True,
        save_vecnormalize=True,
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
        verbose=1
    )
    
    # Training with proper timestep management[3]
    try:
        logging.info(f"Starting training for {total_timesteps} timesteps ({num_of_episodes} episodes)")
        model.learn(total_timesteps=total_timesteps,
                    callback=[checkpoint_cb, eval_cb],
                    progress_bar=True,
                    reset_num_timesteps=False)
        
        model.save(os.path.abspath(f"./rl_models/{model_name}/{model_name}.zip"))
        logging.info("Training completed successfully")
        
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        logging.error(f"Error during training: {e}")
    finally:
        train_env.close()
        env_eval.close()

def test_model(algorithm, reward_function):
    """Test a trained RL model with comprehensive evaluation."""
    # model_name = f"{algorithm}_{reward_function}"
    model_name = algorithm

    # Load the best model
    try:
        model_path = f"rl_models/{model_name}/best_model"
        if algorithm == "DQN":
            model = DQN.load(model_path)
        elif algorithm == "A2C":
            model = A2C.load(model_path)
        elif algorithm == "PPO":
            model = PPO.load(model_path)
    except FileNotFoundError:
        logging.warning(f"Best model not found, loading checkpoint...")
        # Load latest checkpoint
        checkpoint_path = f"rl_models/{model_name}/rl_model_{model_name}_final.zip"
        if algorithm == "DQN":            model = DQN.load(checkpoint_path)
        elif algorithm == "A2C":
            model = A2C.load(checkpoint_path)
        elif algorithm == "PPO":
            model = PPO.load(checkpoint_path)
    
    # Test environment setup
    env = TrafficEnv(port=base_eval_sumo_port,
                     model_name=model_name,
                     model_idx=0,
                     op_mode="test",
                     base_gen_car_distrib=["bimodal", 3],
                     reward_fn=reward_function)
    
    # Evaluation loop
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

def tune_hyperparameters(algorithm, reward_function, n_trials=50):
    """
    Hyperparameter tuning using Optuna for optimal performance.
    
    Args:
        algorithm (str): RL algorithm to tune
        reward_function (str): Reward function type  
        n_trials (int): Number of optimization trials
    """
    
    def objective(trial):
        # Define search space based on algorithm[4][6]
        if algorithm == "DQN":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "buffer_size": trial.suggest_categorical("buffer_size", [50000, 100000, 200000]),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                "target_update_interval": trial.suggest_int("target_update_interval", 1000, 10000),
                "exploration_fraction": trial.suggest_float("exploration_fraction", 0.05, 0.3),
                "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1),
                "gamma": trial.suggest_float("gamma", 0.95, 0.999),
                "net_arch": trial.suggest_categorical("net_arch", 
                    [[256, 256], [512, 256], [256, 256, 128], [512, 256, 128]])
            }
        elif algorithm == "PPO":
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
                "gamma": trial.suggest_float("gamma", 0.95, 0.999),
                "net_arch": trial.suggest_categorical("net_arch", 
                    [[256, 256], [256, 128], [128, 128]])
            }
        
        # Train model with suggested parameters
        try:
            # Use shorter episodes for tuning to save time[4]
            reward = train_model(algorithm, reward_function, 
                               num_of_episodes=2, 
                               use_enhanced_params=False,
                               custom_params=params)
            return reward
        except Exception as e:
            logging.error(f"Trial failed: {e}")
            return float('-inf')
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    logging.info(f"Best parameters: {study.best_params}")
    logging.info(f"Best value: {study.best_value}")
    
    return study.best_params

def get_optimal_params(algorithm, traffic_density, episode_length):
    """
    Select optimal parameters based on traffic conditions and training requirements.
    
    Args:
        algorithm (str): RL algorithm
        traffic_density (str): Expected traffic density ("low", "medium", "high")
        episode_length (str): Training episode length ("short", "medium", "long")
    """
    base_params = ENHANCED_HYPERPARAMS[algorithm].copy()
    
    # Adjust based on traffic density[3][6]
    if traffic_density == "high":
        # More exploration needed for complex scenarios
        if algorithm == "DQN":
            base_params["exploration_fraction"] = 0.2
            base_params["exploration_final_eps"] = 0.05
        base_params["learning_rate"] *= 0.5  # Slower learning for stability
        
    elif traffic_density == "low":
        # Faster convergence possible
        base_params["learning_rate"] *= 1.5
        if algorithm == "DQN":
            base_params["exploration_fraction"] = 0.1
    
    # Adjust based on episode length
    if episode_length == "long":
        base_params["gamma"] = 0.999  # Higher discount for long-term rewards
    elif episode_length == "short":
        base_params["gamma"] = 0.95   # Lower discount for immediate rewards
    
    return base_params

""" Classes """
class TrafficEnv(gym.Env):
    def __init__(self, port, model_name, model_idx, op_mode, base_gen_car_distrib, num_of_episodes=0, reward_fn="balanced"):
        super(TrafficEnv, self).__init__()
        self.default_speed_limit = 130
        self.port = port
        self.model_name = model_name
        self.model_idx = model_idx
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
        
        # Historical data for analysis[1]
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

    def start_sumo(self):
        """Initialize SUMO simulation - only start if not already running properly."""
        # Check if SUMO is already running and responsive
        if self.is_sumo_initialized and self.sumo_process:
            try:
                # Test if SUMO is responsive
                traci.simulation.getTime()
                logging.debug("SUMO is already running and responsive")
                return
            except (FatalTraCIError, TraCIException):
                logging.warning("SUMO process exists but not responsive, restarting...")
                self.is_sumo_initialized = False
        
        # Clean shutdown if needed
        if self.sumo_process and psutil.pid_exists(self.sumo_process.pid):
            self.close_sumo("Restarting SUMO for initialization")
            sleep(2)
        
        for attempt in range(self.sumo_max_retries):
            try:
                port = self.port
                
                # Generate traffic flow
                if self.gen_car_distrib[0] == 'uniform':
                    flow_generation_fix_num_veh(self.model_name, self.model_idx,
                                                self.gen_car_distrib[1],
                                                int(interval_length // 60),
                                                self.num_of_episodes,
                                                num_of_intervals,
                                                self.operation_mode)
                elif self.gen_car_distrib[0] == 'bimodal':
                    flow_generation(self.model_name, self.model_idx,
                                    bimodal_distribution_24h(self.gen_car_distrib[1]), 1)
                
                sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', sumoExecutable)
                self.sumo_process = subprocess.Popen([
                    sumoBinary, "-c",
                    f"./traffic_environment/sumo/3_2_merge_{self.model_name}_{self.model_idx}.sumocfg",
                    '--start',
                    "--default.emergencydecel=7",
                    '--random-depart-offset=3600',
                    "--remote-port", str(port),
                    "--step-length=0.1",
                    "--default.action-step-length=0.2",
                    f"--end={self.sim_length}",  # Set proper end time[2]
                    "--quit-on-end"
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                logging.info(f"Attempting to connect to SUMO on port {port}")
                traci.init(port=port)
                logging.info(f"Successfully connected to SUMO on port {port} for {self.sim_length}s simulation")
                self.is_sumo_initialized = True
                break
                
            except (FatalTraCIError, TraCIException) as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                self.close_sumo("Failed to start SUMO")
                if attempt < self.sumo_max_retries - 1:
                    sleep(2)
                else:
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
        
        # Apply action: gradual speed limit changes[1]
        speed_changes = [-5, 0, +5]
        previous_speed_limit = self.current_speed_limit
        self.current_speed_limit += speed_changes[action]
        
        # Invalid action penalty and clamping
        invalid_action_penalty = 0
        if (previous_speed_limit <= 50 and action == 0) or (previous_speed_limit >= 130 and action == 2):
            invalid_action_penalty = -1
        
        self.current_speed_limit = max(50, min(130, self.current_speed_limit))
        
        # Apply speed limit to controlled segment
        for segId in seg_1_before:
            traci.lane.setMaxSpeed(segId, self.current_speed_limit / 3.6)
        
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
            
            # Collect traffic measurements[1]
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
            
            # Collision detection[1]
            collisions_in_step = traci.simulation.getCollidingVehiclesNumber()
            if collisions_in_step > 0:
                self.collisions.append(current_time)
        
        # Process collected data
        self.avg_speed_before = mean_speeds_downstream / self.aggregation_time
        self.flow_upstream = (flow_upstream_temp / self.aggregation_time) * 3600
        self.flow_downstream = (flow_downstream_temp / self.aggregation_time) * 3600
        self.queue_length_upstream = queue_length_temp / self.aggregation_time
        self.occupancy_upstream = min(occupancy_upstream_temp / self.aggregation_time, 100.0)
        
        # Update historical data for smoothing[1]
        self.flow_downstream_history.append(self.flow_downstream)
        self.occupancy_downstream_history.append(self.occupancy_upstream)
        self.speed_history.append(self.avg_speed_before)
        
        # Calculate smoothed values
        self.flow_smoothed = np.mean(list(self.flow_downstream_history)) if self.flow_downstream_history else 0
        self.occupancy_smoothed = np.mean(list(self.occupancy_downstream_history)) if self.occupancy_downstream_history else 0
        
        # Collision penalty (2-hour sliding window)[1]
        expiration_time = current_time - (2 * 3600)
        self.collisions = [t for t in self.collisions if t > expiration_time]
        self.collisions_penalty = -5 if len(self.collisions) > 2 else 0
        
        # Calculate reward
        reward = self._calculate_reward(invalid_action_penalty)
        
        # Prepare observation
        speed_trend_val = self._calculate_speed_trend()
        observation = np.array([
            self.avg_speed_before,
            self.flow_upstream,
            self.flow_smoothed,
            self.queue_length_upstream,
            speed_trend_val,
            self.occupancy_smoothed / 100.0,
            self.current_speed_limit
        ], dtype=np.float64)
        
        # Check termination conditions
        # End when simulation time reaches limit OR no more vehicles expected
        done = (current_time >= self.sim_length) or (traci.simulation.getMinExpectedNumber() <= 0)
        
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
        
        # Initial observation
        observation = np.array([
            self.default_speed_limit / 3.6,
            0.0, 0.0, 0.0, 0.0, 0.0,
            self.default_speed_limit
        ], dtype=np.float64)
        
        info = {
            'flow_upstream': 0, 'flow_downstream': 0, 'occupancy': 0,
            'queue_length': 0, 'speed_limit': self.default_speed_limit,
            'collisions': 0, 'simulation_time': 0, 'simulation_step': 0
        }
        
        return observation, info

    def close_sumo(self, reason):
        """Properly close SUMO simulation."""
        logging.debug(f"Closing SUMO: {reason}")
        self.logger.save_to_csv(
            f"rl_train_{self.model_name}_{self.model_idx}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        )
        
        if self.is_sumo_initialized:
            try:
                traci.close()
                logging.debug(f"SUMO closed: {reason}")
            except (FatalTraCIError, TraCIException):
                logging.debug("SUMO was already closed")
            finally:
                self.is_sumo_initialized = False
                self.sumo_process = None

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

if __name__ == '__main__':
    # Suppress matplotlib debug output
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    # from https://sumo.dlr.de/docs/TraCI/Interfacing_TraCI_from_Python.html
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        logging.info("SUMO environment is not set up correctly.")

    option = 3
    algo_used = "DQN"
    reward_used = "balanced"
    create_sumocfg(algo_used)

    if option == 1:
        # Option 1: Use enhanced parameters directly
        train_model(algorithm=algo_used, reward_function=reward_used, use_enhanced_params=True)
    elif option == 2:
        # Option 2: Use adaptive parameter selection
        optimal_params = get_optimal_params(algorithm=algo_used, traffic_density="high", episode_length="long")
        train_model(algorithm=algo_used, 
                    reward_function=reward_used,
                    use_enhanced_params=False,
                    custom_params=optimal_params)
    elif option == 3:
        # Option 3: Run hyperparameter tuning first
        best_params = tune_hyperparameters(algorithm=algo_used, reward_function=reward_used, n_trials=30)
        train_model(algorithm=algo_used, 
                    reward_function=reward_used,
                    use_enhanced_params=False,
                    custom_params=best_params)

    # test_model(algorithm=algo_used, reward_function=reward_used)