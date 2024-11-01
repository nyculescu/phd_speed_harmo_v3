# inspired by https://github.com/nicknochnack/OpenAI-Reinforcement-Learning-with-Custom-Environment

import gymnasium as gym
import numpy as np
import traci
from traci.exceptions import FatalTraCIError, TraCIException
import subprocess
import psutil
import logging
logging.basicConfig(level=logging.DEBUG) # FIXME: use WARNING for long runs. More info at: https://docs.python.org/3/library/logging.html#logging-levels
from time import sleep

from traffic_environment.reward_functions import *
# from rl_utilities.model import *
from traffic_environment.road import *
from traffic_environment.setup import *
from traffic_environment.flow_gen import *

# Initialize counters for vehicle types
vehicle_counts = {
    "normal_car": 0,
    "fast_car": 0,
    "van": 0,
    "bus": 0,
    "motorcycle": 0,
    "truck": 0
}

class TrafficEnv(gym.Env):
    def __init__(self, port, model, model_idx):
        super(TrafficEnv, self).__init__()
        self.port = port
        self.model = model
        self.model_idx = model_idx
        # Actions will be one of the following values [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
        self.speed_limits = np.array([30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
        if model == "TD3" or model == "SAC":
            self.action_space = gym.spaces.Box(low=np.min(self.speed_limits), high=np.max(self.speed_limits), shape=(1,), dtype=np.float64)
        else:
            self.action_space = gym.spaces.Discrete(len(self.speed_limits))
        self.obs = np.array([0], dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([130/3.6]), dtype="float64", shape=(1,))
        self.speed_limit = 130 # this is changed actively
        self.aggregation_time = 60 # [s] duration over which data is aggregated or averaged in the simulation
        self.sim_length = (car_generation_rates * 60 * 60) / self.aggregation_time
        self.mean_speeds_mps = []
        self.mean_emissions = []
        self.mean_num_halts = []
        self.mean_occupancy = []
        self.occupancy_sum = 0
        self.occupancy_avg = 0
        self.occupancy_curr = 0
        self.occupancy = 0
        self.flow = 0
        self.flows = []
        self.sumo_process = None
        self.sumo_max_retries = 5
        self.flow_gen_max = np.random.triangular(0.45, np.random.uniform(0.9, 1.1), 1.55)
        self.reward = 0
        self.total_emissions_now = 0
        self.avg_speed_now = 0
        self.rewards = []
        self.is_learning = True

    def reward_func_wrap(self):
        return quad_occ_reward(self.occupancy)
        # reward_co2_avgspeed(self.prev_emissions, self.total_emissions_now, self.prev_mean_speed, self.avg_speed_now)

    def start_sumo(self):
        # If SUMO is running, then perform a restart
        is_sumo_running = psutil.pid_exists(self.sumo_process.pid) if self.sumo_process else False
        if is_sumo_running:
            logging.warning("SUMO process is still running. Terminating...")
            self.close_sumo("SUMO closed so that it can be restarted")
            sleep(2)
        
        for attempt in range(self.sumo_max_retries):
            try:
                self.flow_gen_max = self.flow_gen_max * np.random.uniform(0.9, 1.3) # add some deviation
                if self.is_learning:
                    flow_generation_wrapper(self.flow_gen_max, self.model, self.model_idx)
                
                port = self.port
                logging.debug(f"Attempting to start SUMO on port {port}")
                self.sumo_process = subprocess.Popen([sumoBinary, "-c", f"./traffic_environment/sumo/3_2_merge_{self.model}_{self.model_idx}.sumocfg", '--start'] + ["--remote-port", str(port)], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
                logging.debug(f"Attempting to connect to SUMO on port {port}")
                traci.init(port=port)
                logging.info(f"Successfully connected to SUMO on port {port}")
                break
            except (FatalTraCIError, TraCIException) as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt <= self.sumo_max_retries:
                    sleep(2)  # Wait before retrying
                else:
                    self.close_sumo("failed at starting SUMO")
                    raise e  # Re-raise the exception if all retries fail

    def close_sumo(self, reason):
        print(f"Closing SUMO: {reason}")
        if hasattr(self, 'sumo_process') and self.sumo_process is not None:
            for attempt in range(self.sumo_max_retries):
                try:
                    traci.close()
                    sleep(2)
                    logging.debug(f"The reason SUMO is closed: {reason}")
                    break
                except (FatalTraCIError, TraCIException):
                    logging.debug(f"SUMO is not closing. {reason}")
                    if attempt <= self.sumo_max_retries:
                        sleep(2)  # Wait before retrying
                    else:
                        logging.debug("Failed stopping traci. sumo_process will be emptied")
                        self.sumo_process.terminate()
                        self.sumo_process.wait()
                        self.sumo_process = None

    def step(self, action):
        is_sumo_running = psutil.pid_exists(self.sumo_process.pid) if self.sumo_process else False
        if not is_sumo_running:
            self.start_sumo()

        if self.model == "TD3" or self.model == "SAC":
            # Map continuous action to nearest discrete speed limit
            self.speed_limit = self.speed_limits[np.argmin(np.abs(self.speed_limits - action[0]))]
        else:
            # Apply action
            self.speed_limit = self.speed_limits[action]
        
        speed_new_mps = self.speed_limit / 3.6  # km/h to m/s

        for segment in [seg_1_before]:
            if is_sumo_running:
                [traci.lane.setMaxSpeed(lane, speed_new_mps) for lane in segment]

        mean_speeds_edges_mps = np.zeros(len(edges))
        emissions_edges = np.zeros(len(edges))
        veh_time_sum = 0

        # Initialize previous state variables if not already set
        if not hasattr(self, 'prev_emissions'):
            self.prev_emissions = np.zeros(len(edges))
            self.prev_mean_speed = np.zeros(len(edges))

        # Step simulation and collect data
        for i in range(self.aggregation_time):
            try:
                traci.simulationStep()
            except (FatalTraCIError, TraCIException):
                logging.debug("Lost connection to SUMO. Attempting to reconnect...")
                self.close_sumo("lost comm with SUMO")
                return self.reset()[0]*4, 0, True, False, {}  # End episode on connection loss

            veh_time_sum += sum([traci.inductionloop.getLastStepVehicleNumber(loop) for loop in loops_after])

            for idx, edge in enumerate(edges):
                mean_speeds_edges_mps[idx] += traci.edge.getLastStepMeanSpeed(edge)
                emissions_edges[idx] += traci.edge.getCO2Emission(edge)

            occ_max = 0
            for loops in loops_before:
                occ_loop = sum([traci.inductionloop.getLastStepOccupancy(loop) for loop in loops]) / len(loops)
                if occ_loop > occ_max:
                    occ_max = occ_loop
            
            self.occupancy_curr = occ_max
            self.occupancy_sum += occ_max

        # Calculate average speed and emissions over the aggregation period
        avg_speed_now = np.mean(mean_speeds_edges_mps) / self.aggregation_time
        self.total_emissions_now = np.mean(emissions_edges) / self.aggregation_time

        # Update previous state variables
        self.prev_emissions = self.total_emissions_now
        self.prev_mean_speed = mean_speeds_edges_mps / self.aggregation_time

        # Calculate reward using the adapted reward function
        # quad_occ_reward()
        self.reward = self.reward_func_wrap()
        
        self.rewards.append(self.reward)

        flow = (veh_time_sum / self.aggregation_time) * (car_generation_rates * 60 * 60)
        self.flows.append(flow)

        self.occupancy = self.occupancy_sum / self.aggregation_time

        self.sim_length -= 1  # Reduce simulation length by 1 second
        done = self.sim_length <= 0
        if done:
            self.close_sumo("simulation done")
            self.rewards = []

        self.obs = np.array([avg_speed_now])

        return self.obs, self.reward, done, False, {}

    def render(self):
        # Implement viz
        pass
    
    def log_info(self):
        logging.info(f"\n-----------------------------------\n\
                     SUMO PID: {self.sumo_process.pid}")
        logging.info(f"Reward for this episode: {self.reward}")
        logging.info(f"Max CO2 emis: {max(self.total_emissions_now)} mg/s")
        logging.info(f"Max speed prev: {max(self.prev_mean_speed)} m/s")
        logging.info(f"Mean flow: {np.mean(self.flows)}")

    def reset(self, seed=None):
        self.flow_gen_max = 0
        self.isUnstableFlowConditions = False
        self.mean_speeds_mps = []
        self.mean_emissions = []
        self.mean_num_halts = []
        self.mean_occupancy = []
        self.flow = 0
        self.flows = []
        self.total_waiting_time = 0
        # self.sumo_process = None
        self.sumo_max_retries = 5
        self.occupancy_sum = 0
        self.occupancy = 0
        self.occupancy_avg = 0
        self.occupancy_curr = 0
        self.rewards = []
        self.avg_speed_now = 0
        self.total_emissions_now = 0
        self.prev_emissions = 0
        self.prev_mean_speed = 0

        # Reset params
        self.speed_limit = 130
        
        # Reset time
        self.sim_length = (car_generation_rates * 60 * 60) / self.aggregation_time

        obs = np.array([0], dtype=np.float64)
        
        return obs, {}
    
    def close(self):
        self.close_sumo("close() method called")

    def __del__(self):
        try:
            self.close()
        except AttributeError:
            # Handle cases where attributes might not be initialized
            print("Warning: Attempted to delete an uninitialized TrafficEnv instance.")
