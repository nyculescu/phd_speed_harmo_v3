import logging
import logging.handlers
from stable_baselines3 import DQN
import torch
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
import time
import psutil
from time import sleep
import traci
from traci import FatalTraCIError, TraCIException
import subprocess
import sys
from pathlib import Path
import csv
import collections

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
interval_length_h = 1 # hours
num_of_intervals = 1
num_test_envs_per_model = 1
num_train_envs_per_model = 1
num_envs_per_model = num_train_envs_per_model + num_test_envs_per_model
interval_length = 60 * interval_length_h
sumoExecutable = 'sumo-gui.exe' if os.name == 'nt' else 'sumo-gui'
sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', sumoExecutable)

MAX_OCCUPANCY = 100.0  # Occupancy measured 0..100
MAX_FLOW = 7200.0      # 2 lanes, up to 2 cars/sec => 7200 cars/hour (example)
MAX_SPEED_DIFF = 80.0  # Range from 50 to 130 => 80 km/h difference
OBSERVATION_SPACE_SIZE = 7

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
            <lateral-resolution value="0.8"/>
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

def train_env_constructor(idx, model_name, num_of_episodes):
    def _init():
        env = Monitor(TrafficEnv(port=base_train_sumo_port + idx, 
                                 model_name=model_name, 
                                 model_idx=idx, 
                                 op_mode="train", 
                                 base_gen_car_distrib=["uniform", 2000],
                                 num_of_episodes=num_of_episodes))
        return env
    return _init

def eval_env_constructor(model_name):
    def _init():
        env = Monitor(TimeLimit(TrafficEnv(port=base_eval_sumo_port, 
                                           model_name=model_name, 
                                           model_idx=num_envs_per_model - 1, 
                                           op_mode="eval", 
                                           base_gen_car_distrib=["uniform", 3000],
                                           num_of_episodes=1), 
                max_episode_steps=interval_length))
        return env
    return _init

def train_dqn(num_of_episodes):
    total_timesteps = int(interval_length * num_of_intervals * num_of_episodes)
    eval_timesteps = int(interval_length * num_of_intervals)
    model_name = 'DQN'
    log_dir = f"./logs/{model_name}/"
    model_dir = f"./rl_models/{model_name}/"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    train_env = SubprocVecEnv([
        train_env_constructor(i, model_name, num_of_episodes)
        for i in range(num_train_envs_per_model)
    ])
    env_eval = SubprocVecEnv([eval_env_constructor(model_name)])

    # policy_kwargs = dict(
    #     net_arch=[256, 256, 128],  # Deeper network
    #     activation_fn=nn.ReLU
    # )
    
    model = DoubleDQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,
        buffer_size=100000,
        batch_size=128,
        gamma=0.99,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        # policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        device='cuda'
    )
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

    try:
        model.learn(total_timesteps=total_timesteps, 
            callback=[checkpoint_cb, eval_cb], 
            progress_bar=True, 
            reset_num_timesteps=False
            )
        model.save(os.path.abspath(f"./rl_models/{model_name}/{model_name}.zip"))
        
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        logging.error(f"Error during training: {e.args}")
    finally:
        # Clean up environments after training ends
        train_env.close()
        env_eval.close()

def test_dqn():
    model_name = "DQN"
    model = DQN.load(f"rl_models/{model_name}/best_model")
    model.policy.eval()  # Set policy to evaluation mode

    logging.debug(f"Starting {model_name} test")
    env = TrafficEnv(port=base_eval_sumo_port, 
                     model_name=model_name, 
                     model_idx=num_envs_per_model - 1, 
                     op_mode="test", 
                     base_gen_car_distrib=["bimodal", 3])

    tf_logger = configure(f"./tensorboard_logs/{model_name}_test", ["tensorboard"])
    model.set_logger(tf_logger)

    logging.info("Loaded best reward weights for testing.")

    obs, _ = env.reset()
    done = False

    rewards = []
    # Initialize custom callback for logging with the environment passed in
    tensorboard_callback = TensorboardCallback(env, model)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        
        # Log reward and other metrics to TensorBoard using the callback
        tensorboard_callback.locals = {'rewards': reward}
        tensorboard_callback._on_step()  # Manually call _on_step() to log metrics
        
        rewards.append(reward)

    env.close()

    tf_logger.dump(step=0)  # Ensure logs are written

""" Classes """
class TrafficEnv(gym.Env):
    def __init__(self, port, model_name, model_idx, op_mode, base_gen_car_distrib, num_of_episodes = 0):
        super(TrafficEnv, self).__init__()
        self.default_speed_limit = 130 # [km/h]
        self.port = port
        self.model_name = model_name
        self.model_idx = model_idx
        self.aggregation_time = 60 # [s] Data aggregation duration
        self.occupancy_sum = 0
        self.occupancy = 0
        self.sumo_process = None
        self.sumo_max_retries = 3
        self.operation_mode = op_mode
        self.is_first_step_delay_on = False
        self.collisions = 0
        self.gen_car_distrib = base_gen_car_distrib
        self.logger = TrafficDataLogger(self.default_speed_limit)
        self.num_of_episodes = num_of_episodes
        self.state_before = 130
        self.preloaded_weights = [0.5, 0.3, 0.2]

        self.action_space = gym.spaces.Discrete(3) # [-5 km/h, +0 km/h, +5 km/h]
        self.current_speed_limit = self.default_speed_limit
        
        # This is a blueprint, defining what observations can look like
        self.observation_space = spaces.Box(
            low=np.array([0,  0,  0,  0, -np.inf, 0, 50]),
            high=np.array([
                self.default_speed_limit/3.6,  # avg_speed_before
                np.inf,                       # flow_upstream
                np.inf,                       # flow_downstream
                np.inf,                       # queue_length_upstream
                np.inf,                       # speed_trend_val
                1.0,                          # occupancy in fraction form
                130                           # last_speed_limit
            ]),
            shape=(OBSERVATION_SPACE_SIZE,),
            dtype=np.float64
        )

        # Initialize historical data structures
        self.speed_history = collections.deque(maxlen=15)  # For speed trend
        self.queue_length_upstream = 0
        self.flow_upstream = 0
        self.flow_downstream = 0
        self.time_step = 0  # To track simulation time steps
        
        if self.operation_mode == "eval":
            self.sim_length = int(interval_length * num_of_intervals * self.num_of_episodes)
        elif self.operation_mode == "test":
            self.sim_length = int(24 * 60)
    
    def start_sumo(self):
        # If SUMO is running, then perform a restart
        is_sumo_running = psutil.pid_exists(self.sumo_process.pid) if self.sumo_process else False
        if is_sumo_running:
            logging.error("SUMO process is still running. Terminating...")
            self.close_sumo(reason="SUMO closed because start_sumo() was called")
            sleep(2)
        
        for attempt in range(self.sumo_max_retries):
            try:
                port = self.port
                if self.gen_car_distrib[0] == 'uniform':
                    flow_generation_fix_num_veh(self.model_name, self.model_idx, 
                                                self.gen_car_distrib[1], 
                                                int(interval_length // 60), 
                                                self.num_of_episodes, 
                                                num_of_intervals, 
                                                self.operation_mode)
                elif self.gen_car_distrib[0] == 'bimodal':
                    flow_generation(self.model_name, self.model_idx, bimodal_distribution_24h(self.gen_car_distrib[1]), 1)
                sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', sumoExecutable)
                self.sumo_process = subprocess.Popen([sumoBinary, "-c", f"./traffic_environment/sumo/3_2_merge_{self.model_name}_{self.model_idx}.sumocfg", '--start'] 
                                                     + ["--default.emergencydecel=7"]
                                                     + ['--random-depart-offset=3600']
                                                     + ["--remote-port", str(port)] 
                                                     + ["--step-length=0.1"]
                                                     + ["--default.action-step-length=0.1"]
                                                     + ["--quit-on-end"]
                                                       ,
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
                logging.info(f"Attempting to connect to SUMO on port {port}")
                traci.init(port=port)
                logging.info(f"Successfully connected to SUMO on port {port}")
                break
            except (FatalTraCIError, TraCIException) as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                self.close_sumo(reason="failed at starting SUMO. Restarting...")

                if attempt <= self.sumo_max_retries:
                    sleep(2)  # Wait before retrying
                else:
                    self.close_sumo(reason="failed at starting SUMO")
                    raise e  # Re-raise the exception if all retries fail

    def close_sumo(self, reason):
        logging.debug(f"Closing SUMO: {reason}")
        self.logger.save_to_csv(f"rl_test_{self.model_name}_{self.model_idx}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        if hasattr(self, 'sumo_process') and self.sumo_process is not None:
            for attempt in range(self.sumo_max_retries):
                try:
                    traci.close()
                    logging.debug(f"The reason SUMO is closed: {reason}")
                    break
                except (FatalTraCIError, TraCIException):
                    logging.debug(f"SUMO is not closing when {reason}")
                    if attempt <= self.sumo_max_retries:
                        sleep(2)  # Wait before retrying
                        logging.debug("Killing SUMO process due to non-responsive close") 
                finally:
                    self.sumo_process = None
    
    def feedback_adjustment(self, speed_limit):
        # If free-flow, do nothing
        if self.occupancy < 12:
            return speed_limit
        
        speed_variance = np.var(list(self.speed_history))
        # Adjust speed if traffic is unstable or queue is large
        if (self.flow_downstream < self.flow_upstream * 0.8) or (self.queue_length_upstream > 100) or (speed_variance > 5):
            speed_limit = max(speed_limit - 10, 50)
        elif (self.flow_downstream > self.flow_upstream * 1.2) and (self.queue_length_upstream < 50) and (speed_variance < 2):
            speed_limit = min(speed_limit + 10, self.default_speed_limit)
        
        return speed_limit

    def step(self, action):
        is_sumo_running = psutil.pid_exists(self.sumo_process.pid) if self.sumo_process else False
        if not is_sumo_running:
            self.start_sumo()
        else:
            for segment in [seg_1_before]:
                [traci.lane.setMaxSpeed(segId, self.default_speed_limit / 3.6) for segId in segment] # km/h to m/s

        speed_changes = [-5, 0, +5] # Gradual Speed Adjustment instead of Absolute Speed Mapping
        self.current_speed_limit += speed_changes[action]
        invalid_action_penaly = -1 if (self.current_speed_limit == 50 and action == 0) or (self.current_speed_limit == 130 and action == 2) else 0
        self.current_speed_limit = max(50, min(130, self.current_speed_limit)) # state clamping
        state = self.current_speed_limit

        flow_upstream_temp = 0
        flow_downstream_temp = 0
        queue_length_temp = 0
        mean_speeds_seg_1_before = 0
        mean_speeds_seg_0_after = 0
        self.occupancy_sum = 0
        mean_speed_merge_seg = 0
        departed_vehicles = 0

        while self.is_first_step_delay_on:
            try:
                traci.simulationStep()
                if traci.edge.getLastStepVehicleNumber("seg_0_after") > 0:
                    self.is_first_step_delay_on = False
                    break
            except (FatalTraCIError, TraCIException):
                logging.debug("Lost connection to SUMO from first step")
                self.close_sumo("lost comm with SUMO")
                return self.reset()

        # Simulation steps and data collection
        for _ in range(self.aggregation_time):
            try:
                traci.simulationStep()
            except (FatalTraCIError, TraCIException):
                logging.debug("Lost connection to SUMO from steps")
                self.close_sumo("lost comm with SUMO")
                return self.reset()
    
            # Collect flow measurements
            flow_upstream_temp += traci.edge.getLastStepVehicleNumber("seg_1_before")
            flow_downstream_temp += traci.edge.getLastStepVehicleNumber("seg_0_after")
            mean_speed_merge_seg += traci.edge.getLastStepMeanSpeed("seg_0_after")
    
            # Collect queue length upstream based on halting vehicles
            queue_length_temp += sum([
                traci.lane.getLastStepHaltingNumber(lane) * 7.5
                for lane in seg_1_before
            ])
    
            mean_speeds_seg_1_before += traci.edge.getLastStepMeanSpeed("seg_1_before")
            mean_speeds_seg_0_after += traci.edge.getLastStepMeanSpeed("seg_0_after")
    
            # Calculate occupancy
            occ_max = 0
            for loops in loops_before:
                occ_loop = sum([traci.inductionloop.getLastStepOccupancy(loop) for loop in loops]) / len(loops)
                if occ_loop > occ_max:
                    occ_max = occ_loop
            self.occupancy_sum += occ_max

            departed_vehicles += traci.simulation.getDepartedNumber()

        self.occupancy = self.occupancy_sum / self.aggregation_time
        avg_speed_before = mean_speeds_seg_1_before / self.aggregation_time
        
        self.flow_upstream = (flow_upstream_temp * 3600) / self.aggregation_time
        self.flow_downstream = (flow_downstream_temp * 3600) / self.aggregation_time
        self.queue_length_upstream = queue_length_temp / self.aggregation_time
        avg_speed_merging = mean_speed_merge_seg / self.aggregation_time

        self.speed_history.append(avg_speed_before)
        speed_history_queue_length = 10
        if len(self.speed_history) < speed_history_queue_length:
            avg_speed_trend = [avg_speed_before] * speed_history_queue_length
        else:
            # Convert to list if self.speed_history is not already a list
            avg_speed_trend = list(self.speed_history)[-speed_history_queue_length:]
        speed_trend_val = avg_speed_merging - np.mean(avg_speed_trend)

        # Time-step update (e.g., for cyclical encoding) used only in the agent testing
        self.time_step = (self.time_step + 1) % (24 * 60) if self.operation_mode == "test" else 0

        # Apply feedback adjustment based on downstream impact and queue lengths
        state = self.feedback_adjustment(state)

        # Speed Smoothing
        state = math.ceil(int(0.749 * state + 0.251 * self.logger.get_last_logged_speed()) / 5) * 5

        # Apply the adjusted speed limit to all lanes
        [traci.lane.setMaxSpeed(l, state/3.6) for l in seg_1_before]
        
        # This is an instance of self.observation_space, conforming to its blueprint
        observation = np.array([
            avg_speed_before,
            self.flow_upstream,
            self.flow_downstream,
            self.queue_length_upstream,
            speed_trend_val,
            self.occupancy / MAX_OCCUPANCY,
            float(self.state_before)
        ], dtype=np.float64)
        
        # Compute reward dynamically using RewardWeightNet
        R_occ = min(self.occupancy / MAX_OCCUPANCY, 1.0)
        R_flow = min(self.flow_downstream / MAX_FLOW, 1.0)
        # Speed smoothing: penalize large jumps from last action or from default_speed
        #    Example: difference from last_speed_limit => self.current_speed_limit
        #    Scale it to [0..1] by dividing by MAX_SPEED_DIFF, then turn into negative reward
        R_smooth = - (abs(self.current_speed_limit - self.state_before) / MAX_SPEED_DIFF)
        
        if self.operation_mode == "train":
            # obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            # weights = self.reward_weight_net(obs_tensor).detach().numpy().flatten()
            weights = self.preloaded_weights # FIXME: Get a stable framework to generate the best weights
            w_occ, w_flow, w_smooth = weights[0], weights[1], weights[2]
            reward = (w_occ * R_occ) + (w_flow * R_flow) + (w_smooth * R_smooth) + invalid_action_penaly
        else:
            # Load stored weights (preloaded in test_dqn function)
            weights = self.preloaded_weights  # Use preloaded weights
            w_occ, w_flow, w_smooth = weights[0], weights[1], weights[2]
            reward = (w_occ * R_occ) + (w_flow * R_flow) + (w_smooth * R_smooth) + invalid_action_penaly

        # Log metrics
        self.logger.log_step(
            vss=state,
            occupancy=self.occupancy,
            speed_before=avg_speed_before,
            speed_after=mean_speeds_seg_0_after / self.aggregation_time,
            reward=reward,
            flow_upstream=self.flow_upstream,
            flow_downstream=self.flow_downstream,
            avg_speed_trend=avg_speed_trend,
            reward_weights = weights,
            rewards = [R_occ, R_flow, R_smooth],
            action=action,
            departed_vehicles=departed_vehicles
        )

        self.state_before = state
    
        # Determine if the episode is done
        if self.operation_mode == "train":
            done = (traci.simulation.getMinExpectedNumber() <= 0)
        else:
            self.sim_length -= 1
            done = self.sim_length <= 0
    
        return observation, reward, done, False, {}

    def render(self):
        # Implement viz
        pass

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.occupancy_sum = 0
        self.occupancy = 0
        self.is_first_step_delay_on = True
        self.collisions = 0

        initial_observation = np.array([0]*OBSERVATION_SPACE_SIZE, dtype=np.float64)

        return initial_observation, {}
    
    def close(self):
        self.close_sumo("close() method called")

    def __del__(self):
        try:
            self.close()
        except AttributeError:
            # Handle cases where attributes might not be initialized
            logging.warning("Attempted to delete an uninitialized TrafficEnv instance.")

class TrafficDataLogger:
    def __init__(self, default_speed_limit, output_dir="logs/rl_test"):
        self.default_speed_limit = default_speed_limit
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.traffic_data = []
        self.header = [
            "timestamp", "VSS", "occupancy", "avg_speed_before", "avg_speed_after",
            "reward", "flow_upstream", "flow_downstream", "avg_speed_trend", 
            "reward_weights", "rewards", "action", "departed_vehicles"
        ]

    def log_step(self, vss, occupancy, speed_before, speed_after, reward,
                 flow_upstream, flow_downstream, avg_speed_trend, reward_weights,
                 rewards, action, departed_vehicles):
        self.traffic_data.append({
            "timestamp": time.time(),
            "VSS": vss,
            "occupancy": occupancy,
            "avg_speed_before": speed_before,
            "avg_speed_after": speed_after,
            "reward": reward,
            "flow_upstream": flow_upstream,
            "flow_downstream": flow_downstream,
            "avg_speed_trend": avg_speed_trend,
            "reward_weights": reward_weights,
            "rewards": rewards,
            "action": action,
            "departed_vehicles": departed_vehicles,
        })

    def save_to_csv(self, filename="traffic_data.csv"):
        filepath = self.output_dir / filename
        with open(filepath, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.header)
            writer.writeheader()
            writer.writerows(self.traffic_data)

    def get_last_logged_speed(self):
        if len(self.traffic_data) > 0:
            return self.traffic_data[-1]["VSS"]
        return self.default_speed_limit  # default fallback

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
    
class DoubleDQN(DQN):
    def __init__(self, policy, env, **kwargs):
        super().__init__(policy=policy, env=env, **kwargs)
    
    def _target_q_value(self, replay_data):
        with torch.no_grad():
            # Select actions using online network
            next_q_values = self.q_net(replay_data.next_observations)
            next_actions = next_q_values.argmax(dim=1).reshape(-1, 1)
            
            # Evaluate actions using target network
            next_q_values_target = self.q_net_target(replay_data.next_observations)
            target_q_values = next_q_values_target.gather(1, next_actions)
        return target_q_values

"""
class RewardWeightNet(nn.Module):
    def __init__(self, input_dim):
        super(RewardWeightNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # First hidden layer
        self.fc2 = nn.Linear(64, 32)        # Second hidden layer
        self.fc3 = nn.Linear(32, 3)         # Output layer. 3 outputs: w_occ, w_flow, w_smooth

    def forward(self, state_features):
        x = F.relu(self.fc1(state_features))  # Apply ReLU activation
        x = F.relu(self.fc2(x))
        x = self.fc3(x)                      # Raw output (logits)
        weights = F.softmax(x, dim=-1)       # Normalize to sum to 1 (softmax)
        return weights

    def save_weights(self, path):
        logging.debug("Saving the reward weights")
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        logging.debug("Loading the reward weights")
        self.load_state_dict(torch.load(path))
"""
        
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

    create_sumocfg("DQN")
    
    train_dqn(num_of_episodes=7)

    test_dqn()