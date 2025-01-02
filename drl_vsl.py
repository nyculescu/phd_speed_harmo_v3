import logging
import logging.handlers
from stable_baselines3 import DQN
import torch
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
import time
import psutil
from time import sleep
import traci
from traci import FatalTraCIError, TraCIException
import subprocess
import sys
from pathlib import Path

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
interval_length_h = 1 # hours
num_of_intervals = 10
num_of_episodes = 10
num_test_envs_per_model = 1
num_train_envs_per_model = 1
num_envs_per_model = num_train_envs_per_model + num_test_envs_per_model
interval_length = 60 * interval_length_h
test_with_electric = True # default value
test_with_disobedient = True # default value
metrics_to_plot = ['rewards'
                #    , 'obs'
                   , 'emissions'
                   , 'mean speed'
                   , 'flow'
                   , 'density'
                   ]
sumoExecutable = 'sumo-gui.exe' if os.name == 'nt' else 'sumo-gui'
sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', sumoExecutable)


def create_sumocfg(model):
    sumocfg_template = """<?xml version="1.0" encoding="UTF-8"?>
    <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
        <input>
            <net-file value="3_2_merge.net.xml"/>
            <route-files value="generated_flows_{model}_{index}.rou.xml"/>
            <additional-files value="loops_detectors.add.xml"/>
            <gui-settings-file value="colored.view.xml"/>
        </input>
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

def train_dqn():
    total_timesteps = int(interval_length * num_of_intervals * num_of_episodes)
    eval_timesteps = int(interval_length * num_of_intervals)
    model_name = 'DQN'
    log_dir = f"./logs/{model_name}/"
    model_dir = f"./rl_models/{model_name}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    def train_env_constructor(idx):
        def _init():
            env = Monitor(TrafficEnv(port=base_train_sumo_port + idx, 
                                     model=model_name, 
                                     model_idx=idx, 
                                     op_mode="train", 
                                     base_gen_car_distrib=["uniform", 2000]))
            return env
        return _init

    def eval_env_constructor():
        def _init():
            env = Monitor(TimeLimit(TrafficEnv(port=base_eval_sumo_port, 
                                               model=model_name, 
                                               model_idx=num_envs_per_model - 1, 
                                               op_mode="eval", 
                                               base_gen_car_distrib=["uniform", 3000]), 
                    max_episode_steps=interval_length))
            return env
        return _init
    
    train_env = SubprocVecEnv([
        train_env_constructor(i)
        for i in range(num_train_envs_per_model)
    ])
    env_eval = SubprocVecEnv([eval_env_constructor()])

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,
        buffer_size=50000,
        batch_size=64,
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

    callbacks = [checkpoint_cb, eval_cb]

    try:
        model.learn(total_timesteps=total_timesteps, 
            callback=callbacks, 
            progress_bar=True, 
            reset_num_timesteps=False
            )
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
                     model=model_name, 
                     model_idx=num_envs_per_model, 
                     op_mode="test", 
                     base_gen_car_distrib=["bimodal", 2.5])
    
    tf_logger = configure(f"./tensorboard_logs/{model_name}_test", ["tensorboard"])
    model.set_logger(tf_logger)

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

    env.logger.save_to_csv(f"rl_test_{model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    env.close()

    tf_logger.dump(step=0)  # Ensure logs are written

""" Classes """
class TrafficEnv(gym.Env):
    def __init__(self, port, model, model_idx, op_mode, base_gen_car_distrib):
        super(TrafficEnv, self).__init__()
        self.port = port
        self.model = model
        self.model_idx = model_idx
        # Actions will be one of the following values [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
        self.speed_limits = np.arange(50, 135, 5) # end = max + step
        self.action_space = gym.spaces.Discrete(len(self.speed_limits))
        self.obs = np.array([0], dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([130/3.6]), dtype="float64", shape=(1,))
        self.speed_limit = 130 # this is changed actively
        self.aggregation_time = 60 # [s] duration over which data is aggregated or averaged in the simulation
        self.occupancy_sum = 0
        self.occupancy = 0
        self.sumo_process = None
        self.sumo_max_retries = 3
        self.reward = 0
        self.operation_mode = op_mode
        self.test_with_electric = test_with_electric
        self.test_with_disobedient = test_with_disobedient
        self.is_first_step_delay_on = False
        self.n_before0 = 0
        self.n_before1 = 0
        self.n_after = 0
        self.collisions = 0
        self.gen_car_distrib = base_gen_car_distrib
        self.logger = TrafficDataLogger()
        if self.operation_mode == "eval":
            self.sim_length = int(interval_length * num_of_intervals * num_of_episodes)
        elif self.operation_mode == "test":
            self.sim_length = int(24 * 60)
            
    def quad_occ_reward(self):
        ''' This function rewards occupancy rates around 12% the most, with decreasing rewards for both lower and higher occupancies.
            - low occupancy rates (0-12%) => increases from 0.5 at x=0 to 1 at x=12
            - medium to high occupancy rates (12-80%) => decreases from 1 at x=12 to 0 at x=80
            - very low (≤0) or very high (≥80) occupancy rates => reward is 0
            The function is continuous at x=12, where both pieces evaluate to 1. However, it is not differentiable at x=12 due to the change in slope.
            Ref.: Kidando, E., Moses, R., Ozguven, E. E., & Sando, T. (2017). Evaluating traffic congestion using the traffic occupancy and speed distribution relationship: An application of Bayesian Dirichlet process mixtures of generalized linear model. Journal of Transportation Technologies, 7(03), 318.
        '''
        if 0 < self.occupancy <= 12:
            return ((0.5 * self.occupancy) + 6) / 12
        elif 12 < self.occupancy < 80:
            return ((self.occupancy-80)**2/68**2)
        else:
            return 0

    def get_collisions_on_segment(self, segment_id):
        # Get the list of vehicles involved in collisions during this simulation step
        colliding_vehicles = traci.simulation.getCollidingVehiclesIDList()
        
        # Initialize a counter for collisions on the specific segment
        collision_count = 0
        
        # Iterate over each colliding vehicle
        for vehicle_id in colliding_vehicles:
            # Get the current road (edge) ID where the vehicle is located
            road_id = traci.vehicle.getRoadID(vehicle_id)
            
            # Check if the road ID matches the target segment (e.g., "seg_0_before")
            if road_id == segment_id:
                collision_count += 1
        
        return collision_count

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
                    flow_generation_fix_num_veh(self.model, self.model_idx, 
                                                self.gen_car_distrib[1], 
                                                int(interval_length // 60), 
                                                num_of_episodes, 
                                                num_of_intervals, 
                                                self.operation_mode)
                elif self.gen_car_distrib[0] == 'bimodal':
                    flow_generation(self.model, self.model_idx, bimodal_distribution_24h(self.gen_car_distrib[1]), 1)
                sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', sumoExecutable)
                self.sumo_process = subprocess.Popen([sumoBinary, "-c", f"./traffic_environment/sumo/3_2_merge_{self.model}_{self.model_idx}.sumocfg", '--start'] 
                                                     + ["--default.emergencydecel=7"]
                                                     + ['--random-depart-offset=3600']
                                                     + ["--remote-port", str(port)] 
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
    
    def step(self, action):
        is_sumo_running = psutil.pid_exists(self.sumo_process.pid) if self.sumo_process else False
        if not is_sumo_running:
            self.start_sumo()
        else:
            for segment in [seg_1_before]:
                [traci.lane.setMaxSpeed(segId, self.speed_limit / 3.6) for segId in segment] # km/h to m/s

        self.speed_limit = self.speed_limits[action]

        veh_time_sum = 0
        num_vehicles_in_merging_zone_temp = 0
        mean_speeds_seg_1_before_mps_temp = 0
        mean_speeds_seg_0_after_mps_temp = 0
        emissions_seg_1_before_temp = 0

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

        n_before0_avg = []
        n_before1_avg = []
        n_after_avg = []
        for i in range(self.aggregation_time):
            if not self.operation_mode == "train" and not self.test_with_electric:
                # Identify vehicles of type "electric_passenger" and change their emission class to match the corresponding petrol vehicle
                vehicle_ids = traci.vehicle.getIDList()
                for vehicle_id in vehicle_ids:
                    veh_id = traci.vehicle.getTypeID(vehicle_id)
                    if veh_id == "electric_passenger":
                        traci.vehicle.setEmissionClass(vehicle_id, "HBEFA4/PC_petrol_Euro-4")
                    if veh_id == "electric_passenger/hatchback" or veh_id == "electric_passenger/van":
                        traci.vehicle.setEmissionClass(vehicle_id, "HBEFA4/PC_petrol_Euro-5")
                    if veh_id == "electric_bus" or veh_id == "electric_truck" or veh_id == "electric_truck/trailer":
                        traci.vehicle.setEmissionClass(vehicle_id, "HBEFA4/RT_le7.5t_Euro-VI_A-C")
                    if veh_id == "electric_motorcycle":
                        traci.vehicle.setEmissionClass(vehicle_id, "HBEFA4/PC_petrol_Euro-6ab")
            if not self.operation_mode == "train" and not self.test_with_disobedient:
                # Identify vehicles of type "disobedient"
                vehicle_ids = traci.vehicle.getIDList()
                for vehicle_id in vehicle_ids:
                    veh_id = traci.vehicle.getTypeID(vehicle_id)
                    if "disobedient" in veh_id:
                        traci.vehicle.setImperfection(vehicle_id, 0.1)
                        traci.vehicle.setImpatience(vehicle_id, 0.1)

            try:
                traci.simulationStep()
            except (FatalTraCIError, TraCIException):
                logging.debug("Lost connection to SUMO from steps")
                self.close_sumo("lost comm with SUMO")
                return self.reset()  # End episode on connection loss

            veh_time_sum += sum([traci.inductionloop.getLastStepVehicleNumber(loop) for loop in loops_after])

            mean_speeds_seg_1_before_mps_temp += traci.edge.getLastStepMeanSpeed("seg_1_before")
            mean_speeds_seg_0_after_mps_temp += traci.edge.getLastStepMeanSpeed("seg_0_after")
            emissions_seg_1_before_temp += traci.edge.getCO2Emission("seg_1_before")
            num_vehicles_in_merging_zone_temp += traci.edge.getLastStepVehicleNumber("seg_0_before")

            occ_max = 0
            for loops in loops_before:
                occ_loop = sum([traci.inductionloop.getLastStepOccupancy(loop) for loop in loops]) / len(loops)
                if occ_loop > occ_max:
                    occ_max = occ_loop
            
            self.occupancy_sum += occ_max

            # self.collisions += len(traci.simulation.getCollidingVehiclesIDList())
            self.collisions += self.get_collisions_on_segment("seg_0_before")
            
            n_before0_avg.append(traci.edge.getLastStepVehicleNumber("seg_0_before"))
            n_before1_avg.append(traci.edge.getLastStepVehicleNumber("seg_1_before"))
            n_after_avg.append(traci.edge.getLastStepVehicleNumber("seg_0_after") + traci.edge.getLastStepVehicleNumber("seg_1_after"))
        
        # Calculate average speed and emissions over the aggregation period
        avg_speed_before_1_now = np.mean(mean_speeds_seg_1_before_mps_temp) / self.aggregation_time
        avg_speed_after_0_now = np.mean(mean_speeds_seg_0_after_mps_temp) / self.aggregation_time

        self.n_before0 = np.average(n_before0_avg) if len(n_before0_avg) > 0 else 0
        self.n_before1 = np.average(n_before1_avg) if len(n_before1_avg) > 0 else 0
        self.n_after = np.average(n_after_avg) if len(n_after_avg) > 0 else 0
        
        self.occupancy = self.occupancy_sum / self.aggregation_time

        self.obs = np.array([avg_speed_before_1_now])

        self.logger.log_step(
            vss=self.speed_limit,
            occupancy=self.occupancy,
            speed_before=avg_speed_before_1_now,
            speed_after=avg_speed_after_0_now,
            reward=self.reward,
            flow_downstream=0,
            flow_upstream=0,
            avg_speed_trend=0
        )

        self.reward = self.quad_occ_reward()

        if self.operation_mode == "train":
            done = (traci.simulation.getMinExpectedNumber() <= 0)
        else:
            self.sim_length -= 1
            if self.sim_length <= 0: 
                done = True
                logging.debug("Sim length elapsed.")
            else:
                done = False

        return self.obs, self.reward, done, False, {}

    def render(self):
        # Implement viz
        pass

    def reset(self, seed=None):
        self.isUnstableFlowConditions = False
        self.total_waiting_time = 0
        self.occupancy_sum = 0
        self.occupancy = 0
        self.is_first_step_delay_on = True
        self.n_before0 = 0
        self.n_before1 = 0
        self.n_after = 0
        self.collisions = 0
        self.speed_limit = 130
        self.obs = np.array([0], dtype=np.float64)
        
        return self.obs, {}
    
    def close(self):
        self.close_sumo("close() method called")

    def __del__(self):
        try:
            self.close()
        except AttributeError:
            # Handle cases where attributes might not be initialized
            logging.warning("Attempted to delete an uninitialized TrafficEnv instance.")

class TrafficDataLogger:
    def __init__(self, output_dir="logs/rl_test"):
        # Initialize the traffic data logger.
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.traffic_data = []
        self.header = ["timestamp", "VSS", "occupancy", "avg_speed_before", "avg_speed_after", "reward", 
                       "flow_upstream", "flow_downstream", "avg_speed_trend"]
        
    def log_step(self, vss, occupancy, speed_before, speed_after, reward, flow_upstream, flow_downstream, avg_speed_trend):
        # Log data from a single simulation step.
        self.traffic_data.append({
            "timestamp": time.time(),
            "VSS": vss,
            "occupancy": occupancy,
            "avg_speed_before": speed_before,
            "avg_speed_after": speed_after,
            "reward": reward,
            "flow_upstream": flow_upstream,
            "flow_downstream": flow_downstream,
            "avg_speed_trend": avg_speed_trend
        })
        
    def save_to_csv(self, filename="traffic_data.csv"):
        # Save logged data to CSV file.
        filepath = self.output_dir / filename
        with open(filepath, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.header)
            writer.writeheader()
            writer.writerows(self.traffic_data)

class TensorboardCallback(BaseCallback):
    def __init__(self, env, model, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.env = env  # Store the environment
        self.model = model  # Store the model

    def _on_step(self) -> bool:
        # Access metrics from the environment
        reward = self.locals['rewards']  # Access rewards from the environment
        emissions = self.env.emissions_over_time[-1]  # Log latest emissions data
        mean_speed = self.env.mean_speed_over_time[-1]  # Log latest mean speed data
        flow = self.env.flows[-1]  # Log latest flow data
        
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
    
    train_dqn()

    # test_dqn()