# inspired by https://github.com/nicknochnack/OpenAI-Reinforcement-Learning-with-Custom-Environment

import gymnasium as gym
import numpy as np
import traci
from traci.exceptions import FatalTraCIError, TraCIException
import subprocess
import psutil
import logging
logging.basicConfig(level=logging.DEBUG)

from rl_utilities.reward_functions import *
# from rl_utilities.model import *
from simulation_utilities.road import *
from simulation_utilities.setup import *

class SUMOEnv(gym.Env):
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = gym.spaces.Discrete(3)
        # avg speed array
        self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([1]))
        self.state = 0 # occupancy
        self.speed_limit = 120 # to be changed actively
        # Set simulation length
        self.aggregation_time = 30
        self.sim_length = 3600/self.aggregation_time # 30 x 120 = 3600 steps        

        #self.reward_func = scipy.stats.norm(105, 7.5).pdf
        self.reward_func = quad_occ_reward

        self.mean_speeds = []
        self.flows = []
        self.emissions_over_time = []

        self.cvs_seg_time = []
        for i in range(len(all_segments)):
            self.cvs_seg_time.append([])
        
        self.sumo_process = None
        self.sumo_max_retries = 5
        self.start_sumo()
    
    def is_sumo_running(self):
        return psutil.pid_exists(self.sumo_process.pid) if self.sumo_process else False

    def start_sumo(self):
        if self.is_sumo_running():
            logging.warning("SUMO process is still running. Terminating...")
            self.close_sumo()
        
        for attempt in range(self.sumo_max_retries):
            try:
                port = 8813 + attempt
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
                self.close_sumo()
        
        raise RuntimeError("Failed to start SUMO after multiple attempts")

    def close_sumo(self):
        if self.sumo_process:
            try:
                traci.close()
            except (FatalTraCIError, TraCIException):
                pass  # Ignore errors when closing
            self.sumo_process.terminate()
            self.sumo_process = None
        
    def step(self, action):
        # Apply action
        # 0 -1 = -1 
        # 1 -1 = 0 
        # 2 -1 = 1 
        self.speed_limit += (action - 1) * 10
      
        # only cap the speed using lower bounds to let the algorithm learn itself
        if self.speed_limit < 10:
            self.speed_limit = 10       
        

        # copied from mtfc
        speed_new = self.speed_limit / 3.6 # km/h to m/s
        road_segments = [seg_1_before]

        for segment in road_segments:
           [traci.lane.setMaxSpeed(lane, speed_new) for lane in segment]
        
        # Calculate reward
        reward = self.reward_func(self.state)
        
        # ------------------------------ SUMO ------------------------------

        # the only relevant parameter until now
        mean_edge_speed = np.zeros(len(edges))

        # set accumulators
        veh_time_sum = 0        

        # simulate one step in SUMO to get new state
        occupancy_sum = 0

        for i in range(self.aggregation_time):
            try:
                traci.simulationStep() 
            except (FatalTraCIError, TraCIException):
                print("Lost connection to SUMO. Attempting to reconnect...")
                self.start_sumo()
                return self.reset()[0], 0, True, False, {}  # End episode on connection loss
                    
            # GATHER METRICS FROM SENSORS    
            # for some it is important to average over the number of lanes on the edge

            # AFTER
            veh_time_sum += sum([traci.inductionloop.getLastStepVehicleNumber(loop) for loop in loops_after])

            emission_sum = 0
            for i, edge in enumerate(edges):
                mean_edge_speed[i] += traci.edge.getLastStepMeanSpeed(edge)
                emission_sum += traci.edge.getCO2Emission(edge)
            self.emissions_over_time.append(emission_sum)

            # BEFORE

            # collecting the number of vehicles and occupancy right in front (or even in) of the merge area
            # choose max occupancy of a few sensors            
            occ_max = 0
            for loops in loops_before:
                occ_loop = sum([traci.inductionloop.getLastStepOccupancy(loop) for loop in loops]) / len(loops)
                if occ_loop > occ_max:
                    occ_max = occ_loop

            occupancy_sum += occ_max
        
        # monitor the safety of road segments (CVS) - stores cvs value for each segment for each aggregation time step
        for i, seg in enumerate(all_segments):
            cvs_sum = 0
            for lane in seg:
                # for cvs
                ids = traci.lane.getLastStepVehicleIDs(lane)
                speeds = []
                for id in ids:
                    speeds.append(traci.vehicle.getSpeed(id))
                speeds = np.array(speeds)
                lane_avg = np.mean(speeds)
                lane_stdv = np.std(speeds)
                if not np.isnan(lane_stdv) and not np.isnan(lane_avg) and lane_avg != 0:
                    cvs_sum += lane_stdv / lane_avg
            cvs_seg = cvs_sum / len(seg)
            if np.isnan(cvs_seg):
                cvs_seg = 0
            self.cvs_seg_time[i].append(cvs_seg)

        # collected metrics are devided by the aggregation time to get the average values
        # OVERALL
        mean_edge_speed = mean_edge_speed / self.aggregation_time # first is acutally a sum
        mean_road_speed = sum(mean_edge_speed) / len(mean_edge_speed)
        self.mean_speeds.append(mean_road_speed) 

        # AFTER THE MERGE
        flow = (veh_time_sum / self.aggregation_time) * 3600
        self.flows.append(flow)     

        occupancy = occupancy_sum / self.aggregation_time      

        # ------------------------------ SUMO ------------------------------
        
        # gets the avg speed from the simulation
        # normalize the speed
        self.state = occupancy
        self.state_speed = mean_road_speed * 3.6  # m/s to km/h
        
        # Multi-dimensional observation
        # obs = np.array([
        #     self.state,
        #     self.state_speed / 120.0,
        #     self.flows[-1] / 3600.0,
        #     self.cvs_seg_time[-1][-1],
        #     self.speed_limit / 120.0
        # ])

        obs = np.array([self.state], dtype="float32") # Occupancy
        
        # Reduce simulation length by 1 second
        self.sim_length -= 1 
        
        # Check if shower is done
        if self.sim_length <= 0: 
            done = True
            self.close_sumo()
        else:
            done = False

        # Return step information
        return obs, reward, done, False, {}

    def render(self):
        # Implement viz
        pass
    
    def reset(self, seed=None):
        # self.close_sumo()
        # self.start_sumo()

        self.mean_speeds = []

        # Reset params
        self.state = 0
        self.state_speed = 0
        self.speed_limit = 120
        # Reset time
        self.sim_length = 3600/self.aggregation_time

        # Reset SUMO
        # traci.close(False)
        # traci.start(sumoCmd)
        self.state = np.array([self.state],dtype="float32")
        print("Reset observation shape:", self.state.shape)

        return self.state, {}
    
    def close(self):
        self.close_sumo()

    def __del__(self):
        self.close()