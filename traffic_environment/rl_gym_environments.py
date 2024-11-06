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
from config import *

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
    def __init__(self, port, model, model_idx, is_learning):
        super(TrafficEnv, self).__init__()
        self.port = port
        self.model = model
        self.model_idx = model_idx
        # Actions will be one of the following values [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
        self.speed_limits = np.array([30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130])
        if model in cont_act_space_models:
            self.action_space = gym.spaces.Box(low=np.min(self.speed_limits), high=np.max(self.speed_limits), shape=(1,), dtype=np.float64)
        else:
            self.action_space = gym.spaces.Discrete(len(self.speed_limits))
        self.obs = np.array([0], dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([130/3.6]), dtype="float64", shape=(1,))
        self.speed_limit = 130 # this is changed actively
        self.aggregation_time = 60 # [s] duration over which data is aggregated or averaged in the simulation
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
        self.reward = 0
        self.total_emissions_now = 0
        self.avg_speed_now = 0
        self.rewards = []
        self.is_learning = is_learning
        self.emissions_over_time = []
        self.mean_speed_over_time = []
        self.test_without_electric = False
        self.test_without_disobedient = False
        self.frictionValue = 1.0
        self.road_condition = "dry"  # Initial condition
        self.frictionValue = None
        self.adjustment_rate = 0
        self.target_friction = None
        self.target_min = None
        self.target_max = None

        if self.is_learning:
            self.sim_length = round((car_generation_rates * 60 * 60) / self.aggregation_time)
        else:
            self.sim_length = round(len(mock_daily_pattern) / 2 * 60 * 60 / self.aggregation_time)

    def reward_func_wrap(self):
        return quad_occ_reward(self.occupancy)
        # reward_co2_avgspeed(self.prev_emissions, self.total_emissions_now, self.prev_mean_speed, self.avg_speed_now)
    
    def calc_friction(self, desired_transition_steps):
        # Define target ranges based on road conditions
        if self.road_condition == "dry":
            target_min, target_max = 0.75, 1.0
        elif self.road_condition == "wet":
            target_min, target_max = 0.5, 0.75
        elif self.road_condition == "icy":
            target_min, target_max = 0.1, 0.5
        else:
            # Default to dry conditions if unspecified
            target_min, target_max = 0.75, 1.0

        # Select a target friction value within the target range
        target_friction = np.random.uniform(target_min, target_max)

        # Initialize frictionValue if it doesn't exist
        if not hasattr(self, 'frictionValue'):
            self.frictionValue = target_friction
            self.adjustment_rate = 0  # No adjustment needed initially
        else:
            # Calculate the adjustment rate dynamically
            friction_difference = target_friction - self.frictionValue

            # Prevent division by zero
            if desired_transition_steps == 0:
                self.adjustment_rate = friction_difference
            else:
                self.adjustment_rate = friction_difference / desired_transition_steps

            # Store target values
            self.target_friction = target_friction
            self.target_min = target_min
            self.target_max = target_max

    def update_environment(self):
        # Simulate changing road conditions
        conditions = ["dry", "wet", "icy"]
        self.road_condition = np.random.choice(conditions)

        # Transition time in steps (e.g., over 2 hours)
        desired_transition_time_hours = 2
        steps_per_second = 60
        desired_transition_steps = desired_transition_time_hours * 3600 * steps_per_second

        self.calc_friction(desired_transition_steps)
        print(f"Road condition: {self.road_condition}, Target friction: {self.target_friction}")

    def start_sumo(self):
        # If SUMO is running, then perform a restart
        is_sumo_running = psutil.pid_exists(self.sumo_process.pid) if self.sumo_process else False
        if is_sumo_running:
            logging.warning("SUMO process is still running. Terminating...")
            self.close_sumo("SUMO closed so that it can be restarted")
            sleep(2)
        
        for attempt in range(self.sumo_max_retries):
            try:
                days_to_run_the_simu = 7 if self.is_learning else 1
                flow_generation_wrapper(np.random.triangular(-0.5, 0, 0.5), self.model, self.model_idx, days=days_to_run_the_simu)
                
                port = self.port
                sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', sumoExecutable)
                self.sumo_process = subprocess.Popen([sumoBinary, "-c", f"./traffic_environment/sumo/3_2_merge_{self.model}_{self.model_idx}.sumocfg", '--start'] 
                                                     + ["--remote-port", str(port)] + ["--quit-on-end"], 
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
                    logging.debug(f"The reason SUMO is closed: {reason}")
                    break
                except (FatalTraCIError, TraCIException):
                    logging.debug(f"SUMO is not closing when {reason}")
                    if attempt <= self.sumo_max_retries:
                        sleep(2)  # Wait before retrying
                        logging.debug("Killing SUMO process due to non-responsive close") 
                        # self.sumo_process.kill()
                finally:
                    self.sumo_process = None               

    def step(self, action):
        # # Update frictionValue incrementally
        # self.frictionValue += self.adjustment_rate

        # # Ensure frictionValue stays within the target range
        # if self.frictionValue > self.target_max:
        #     self.frictionValue = self.target_max
        # elif self.frictionValue < self.target_min:
        #     self.frictionValue = self.target_min

        self.frictionValue = 1.0

        is_sumo_running = psutil.pid_exists(self.sumo_process.pid) if self.sumo_process else False
        if not is_sumo_running:
            self.start_sumo()
        else:
            for segment in [seg_1_before]:
                [traci.lane.setMaxSpeed(segId, self.speed_limit / 3.6) for segId in segment] # km/h to m/s
            [traci.edge.setFriction(edgeId, self.frictionValue) for edgeId in edges]

        if self.model in cont_act_space_models:
            # Map continuous action to nearest discrete speed limit
            self.speed_limit = self.speed_limits[np.argmin(np.abs(self.speed_limits - action[0]))]
        else:
            # Apply action
            self.speed_limit = self.speed_limits[action]

        mean_speeds_edges_mps = np.zeros(len(edges))
        emissions_edges = np.zeros(len(edges))
        veh_time_sum = 0

        # Step simulation and collect data
        for i in range(self.aggregation_time):
            if not self.is_learning and self.test_without_electric:
                # Identify vehicles of type "electric_passenger"
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
            if not self.is_learning and self.test_without_disobedient:
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
        self.total_emissions_now = np.sum(emissions_edges) / self.aggregation_time

        # Update previous state variables
        self.emissions_over_time.append(self.total_emissions_now)
        self.mean_speed_over_time.append(avg_speed_now * 3.6)  # m/s to km/h

        # Calculate reward
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

        # if self.sim_length % 60 == 0:
        #     self.hour += 1
        # if self.hour % 24 == 0:
        #     self.day += 1
        #     self.hour = 0

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
        logging.info(f"Max speed prev: {max(self.mean_speed_over_time)} m/s")
        logging.info(f"Mean flow: {np.mean(self.flows)}")

    def reset(self, seed=None):
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
        self.emissions_over_time = []
        self.mean_speed_over_time = []
        self.frictionValue = 1.0 # Dry road: friction = 1.0; Wet road: friction = 0.7; Icy road: friction = 0.3

        # Reset params
        self.speed_limit = 130
        
        # Reset time
        if self.is_learning:
            self.sim_length = round((car_generation_rates * 60 * 60) / self.aggregation_time)
        else:
            self.sim_length = round(len(mock_daily_pattern) / 2 * 60 * 60 / self.aggregation_time)

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
