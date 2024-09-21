# inspired by https://github.com/nicknochnack/OpenAI-Reinforcement-Learning-with-Custom-Environment

import gymnasium as gym
import numpy as np
import traci
from traci.exceptions import FatalTraCIError, TraCIException
import subprocess
import psutil
import logging
logging.basicConfig(level=logging.DEBUG) # FIXME: use WARNING for long runs. More info at: https://docs.python.org/3/library/logging.html#logging-levels

from rl_utilities.reward_functions import *
# from rl_utilities.model import *
from simulation_utilities.road import *
from simulation_utilities.setup import *
from simulation_utilities.flow_gen import *

# Initialize counters for vehicle types
vehicle_counts = {
    "normal_car": 0,
    "fast_car": 0,
    "van": 0,
    "bus": 0,
    "motorcycle": 0,
    "truck": 0
}

class SUMOEnv(gym.Env):
    def __init__(self, port):
        self.port = port
        # Actions will be one of the following values [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
        self.action_space = gym.spaces.Discrete(11)
        # self.actions_taken = []
        self.observation_space = gym.spaces.Box(low=np.array([0]*4), high=np.array([1]*4), dtype="float32", shape=(4,))
        self.speed_limit = 130 # this is changed actively
        self.aggregation_time = 60 # [s] duration over which data is aggregated or averaged in the simulation
        # Simulation length for 24 hours (86400 seconds)
        self.sim_length = 24 * 60 * 60 / self.aggregation_time  # 60 x 1440 = 86400 steps
        self.mean_speeds_mps = []
        self.mean_emissions = []
        self.mean_num_halts = []
        # self.flows = []
        # self.emissions_over_time = []
        self.total_travel_time = 0 # [min]
        # self.avg_total_travel_time = []
        # self.total_waiting_time = 0
        # self.avg_total_waiting_time = []
        self.sumo_process = None
        self.sumo_max_retries = 5
        self.day_index = 0
        # self.stop_counts = {} # Dictionary to track stop counts for each vehicle
        # self.previous_speeds = {}
        # self.start_sumo()
    
    def start_sumo(self):
        # If SUMO is running, then perform a restart
        is_sumo_running = psutil.pid_exists(self.sumo_process.pid) if self.sumo_process else False
        if is_sumo_running:
            logging.warning("SUMO process is still running. Terminating...")
            self.close_sumo("restarting SUMO")
        
        for attempt in range(self.sumo_max_retries):
            try:
                port = self.port + attempt
                logging.debug(f"Attempting to start SUMO on port {port}")
                self.sumo_process = subprocess.Popen(sumoCmd + ["--remote-port", str(port)], 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE)
                logging.debug(f"Attempting to connect to SUMO on port {port}")
                traci.init(port=port, numRetries=5)
                logging.info(f"Successfully connected to SUMO on port {port}")
                return
            except (FatalTraCIError, TraCIException) as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                self.close_sumo("failed at starting SUMO")
        
        raise RuntimeError("Failed to start SUMO after multiple attempts")

    def close_sumo(self, reason):
        if self.sumo_process:
            try:
                traci.close()
                logging.debug(f"The reason SUMO is closed: {reason}")
            except (FatalTraCIError, TraCIException):
                pass  # Ignore errors when closing
            self.sumo_process.terminate()
            self.sumo_process = None

    def track_vehicle_times(self, all_segments, departure_times, arrival_times, traci):
        # Create a list of the keys to iterate over
        departure_times_keys = list(departure_times.keys())
        arrival_times_keys = list(arrival_times.keys())

        for idx, seg in enumerate(all_segments):
            for lane in seg:
                ids = traci.lane.getLastStepVehicleIDs(lane)
                
                # Iterate over a copy of the keys
                for id in departure_times_keys:
                    if id not in ids:
                        if id in arrival_times_keys:
                            # Check if both keys exist before accessing them
                            dep_time = departure_times.get(id)
                            arr_time = arrival_times.get(id)
                            if dep_time is not None and arr_time is not None:
                                travel_time = max(0, arr_time - dep_time)
                                self.total_travel_time += travel_time / 60
                                # Remove entries from the original dictionaries
                                del departure_times[id]
                                del arrival_times[id]
                                # logging.debug(f"Removed {id} from departure_times and arrival_times")

                for id in ids:
                    current_time = traci.simulation.getTime()
                    if id in departure_times:
                        arrival_times[id] = current_time
                    else:
                        departure_times[id] = current_time
                        # logging.debug(f"Added {id} to departure_times with time {current_time}")
                    self.total_waiting_time += traci.vehicle.getAccumulatedWaitingTime(id) # Calculate total waiting time for all vehicles still in simulation

    def step(self, action):
        is_sumo_running = psutil.pid_exists(self.sumo_process.pid) if self.sumo_process else False
        if not is_sumo_running:
            self.start_sumo()

        # Apply action
        self.speed_limit = 10 * action + 30
        # if not self.speed_limit in self.actions_taken:
        #     self.actions_taken.append(self.speed_limit)

        # copied from mtfc
        speed_new_mps = self.speed_limit / 3.6 # km/h to m/s
        road_segments = [seg_1_before]

        for segment in road_segments:
            if is_sumo_running:
               [traci.lane.setMaxSpeed(lane, speed_new_mps) for lane in segment]
                
        mean_speeds_edges_mps = np.zeros(len(edges))
        num_halts_edges = np.zeros(len(edges))
        emissions_edges = np.zeros(len(edges))
        veh_time_sum = 0        
        occupancy_sum = 0
        # departure_times = {}
        # arrival_times = {}

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
                num_halts_edges[idx] += traci.edge.getLastStepHaltingNumber(edge)  # Number of vehicles stopped
                emissions_edges += traci.edge.getCO2Emission(edge)
            # self.emissions_over_time.append(emissions_edges)
         
            occ_max = 0
            for loops in loops_before:
                occ_loop = sum([traci.inductionloop.getLastStepOccupancy(loop) for loop in loops]) / len(loops)
                if occ_loop > occ_max:
                    occ_max = occ_loop

            occupancy_sum += occ_max

            # Granularity: lane | The following can be used for granularity: vehicle
            # veh_in_collision = traci.simulation.getCollidingVehiclesNumber() # min 2 in case of one collision
            # self.track_vehicle_times(all_segments, departure_times, arrival_times, traci)
                                        
        # self.avg_total_travel_time.append(self.total_travel_time / self.aggregation_time)
        # self.avg_total_waiting_time.append(self.total_waiting_time / self.aggregation_time)

        # for i, seg in enumerate(all_segments):
        #     cvs_sum = 0
        #     for lane in seg:
        #         # for cvs
        #         ids = traci.lane.getLastStepVehicleIDs(lane)
        #         speeds = []
        #         for id in ids:
        #             speeds.append(traci.vehicle.getSpeed(id))
        #         speeds = np.array(speeds)
        #         lane_avg = np.mean(speeds)
        #         lane_stdv = np.std(speeds)
        #         if not np.isnan(lane_stdv) and not np.isnan(lane_avg) and lane_avg != 0:
        #             cvs_sum += lane_stdv / lane_avg
        #     cvs_seg = cvs_sum / len(seg)
        #     if np.isnan(cvs_seg):
        #         cvs_seg = 0
        #     self.cvs_seg_time[i].append(cvs_seg)

        mean_speeds_edges_mps = mean_speeds_edges_mps / self.aggregation_time
        self.mean_speeds_mps.append(sum(mean_speeds_edges_mps) / len(mean_speeds_edges_mps)) 

        num_halts_edges = num_halts_edges / self.aggregation_time
        self.mean_num_halts.append(sum(num_halts_edges)) 

        emissions_edges = emissions_edges / self.aggregation_time
        self.mean_emissions.append(sum(emissions_edges) / len(emissions_edges)) 

        # flow = (veh_time_sum / self.aggregation_time)
        # self.flows.append(flow)     

        # or Traffic Density: Represents the proportion of the road occupied by vehicles. 
        # It's a critical factor in understanding congestion levels. [based on traffic flow model]
        occupancy = occupancy_sum / self.aggregation_time

        # avg_emissions_over_time = np.mean(self.emissions_over_time)
        # if avg_emissions_over_time > max_emissions_over_time:
        #     logging.debug(f"Avg emis. over {max_emissions_over_time}: {avg_emissions_over_time}")

        # Normalize each component of the observation
        normalized_mean_speed = normalize(np.mean(self.mean_speeds_mps), min_value=0, max_value=130/3.6)
        normalized_occupancy = normalize(occupancy, min_value=0, max_value=100)
        normalized_emissions = normalize(np.mean(self.mean_emissions), min_value=0, max_value=max_emissions)
        normalized_halts = normalize(np.mean(self.mean_num_halts), min_value=0, max_value=10)
        # normalized_waiting_time = normalize(self.total_waiting_time, min_value=0, max_value=3600)
        # normalized_travel_time = normalize(self.total_travel_time, min_value=0, max_value=3600)

        # Construct normalized observation vector
        obs = np.array([
            normalized_mean_speed,
            normalized_occupancy,
            normalized_emissions,
            normalized_halts
            # normalized_waiting_time,
            # normalized_travel_time
        ], dtype=np.float32)
        
        reward = complex_reward(speed_new_mps, np.mean(self.mean_speeds_mps), occupancy, np.mean(self.mean_emissions), np.mean(self.mean_num_halts))

        # Reduce simulation length by 1 second
        self.sim_length -= 1 
        done = self.sim_length <= 0
        if done:
            logging.info(f"--------------------------------\nSUMO PID: {self.sumo_process.pid}")

            logging.info(f"Max CO2 emis: {max(self.mean_emissions)} mg/s")
            logging.info(f"Avg CO2 emis: {np.mean(self.mean_emissions)} mg/s")

            logging.info(f"Max Mean speed: {max(self.mean_speeds_mps)} m/s")
            logging.info(f"Avg Mean speed: {np.mean(self.mean_speeds_mps)} m/s")

            logging.info(f"Max No Halts: {max(self.mean_num_halts)}")
            logging.info(f"Avg No Halts: {np.mean(self.mean_num_halts)}")

            logging.info(f"Occupancy Sum: {occupancy_sum}")
            logging.info(f"Occupancy/60: {occupancy}")

            # logging.info(f"TWT : {self.total_waiting_time / self.total_waiting_time}")
            # logging.info(f"Avg TWT : {np.mean(self.avg_total_waiting_time)}")
            # logging.info(f"TTT: {self.total_travel_time}")
            # logging.info(f"Avg TTT: {np.mean(self.avg_total_travel_time)}")
            # logging.info(f"Action: {self.actions_taken}\n--------------------------------")

            self.close_sumo("simualtion done")

        # Return step information
        return obs, reward, done, False, {}

    def render(self):
        # Implement viz
        pass
    
    def reset(self, seed=None):
        self.day_index = 0
        self.mean_speeds_mps = []
        self.mean_emissions = []
        self.mean_num_halts = []
        # self.flows = []
        # self.emissions_over_time = []
        self.total_travel_time = 0
        self.total_waiting_time = 0
        self.sumo_process = None
        self.sumo_max_retries = 5
        self.day_index = 0
        # self.stop_counts = {} # Dictionary to track stop counts for each vehicle
        # self.previous_speeds = {}

        # Reset params
        self.speed_limit = 130
        
        # Reset time
        self.sim_length = 24 * 60 * 60 / self.aggregation_time

        obs = np.array([0]*4, dtype=np.float32)
        
        return obs, {}
    
    def close(self):
        self.close_sumo("close() method called")

    def __del__(self):
        self.close()