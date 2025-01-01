# inspired by https://github.com/nicknochnack/OpenAI-Reinforcement-Learning-with-Custom-Environment

import gymnasium as gym
import numpy as np
import traci
from traci.exceptions import FatalTraCIError, TraCIException
import subprocess
import psutil
import logging
from pathlib import Path
logging.basicConfig(level=logging.DEBUG) # FIXME: use WARNING for long runs. More info at: https://docs.python.org/3/library/logging.html#logging-levels
from time import sleep
from scipy.optimize import minimize
from traffic_environment.reward_functions import *
# from rl_utilities.model import *
from traffic_environment.road import *
from traffic_environment.setup import *
from traffic_environment.flow_gen import *
from config import *
import collections

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
    def __init__(self, port, model, model_idx, is_learning, base_gen_car_distrib):
        super(TrafficEnv, self).__init__()
        self.port = port
        self.model = model
        self.model_idx = model_idx if is_learning else model_idx + num_envs_per_model + 1
        # Actions will be one of the following values [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
        self.speed_limits = np.arange(50, 135, 5) # end = max + step
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
        self.occupancy_sum = 0
        self.occupancy = 0
        self.flow = 0
        self.flows = []
        self.sumo_process = None
        self.sumo_max_retries = 3
        self.reward = 0
        self.total_emissions_now = 0
        self.avg_speed_now = 0
        self.rewards = []
        self.is_learning = is_learning
        self.emissions_over_time = []
        self.mean_speed_over_time = []
        self.test_with_electric = test_with_electric
        self.test_with_disobedient = test_with_disobedient
        self.frictionValue = 1.0
        self.road_condition = "dry"  # Initial condition
        self.frictionValue = None
        self.adjustment_rate = 0
        self.target_friction = None
        self.target_min = None
        self.target_max = None
        self.is_first_step_delay_on = False
        # self.sim_length = episode_length * num_of_hr_intervals * num_of_iterations * num_envs_per_model
        self.prev_act_spdlim = None
        self.act_spdlim_change_amounts = []
        self.density = []
        self.n_before0 = 0
        self.n_before1 = 0
        self.n_after = 0
        self.collisions = 0
        self.gen_car_distrib = base_gen_car_distrib
        self.logger = TrafficDataLogger()

    def reward_func_wrap(self):
        return quad_occ_reward(self.occupancy)
        # return reward_function_v6(self.model, self.n_before, self.n_after, self.avg_speed_now, self.collisions)
        # return reward_function_v10(self.model, self.n_before0, self.n_before1, self.n_after, self.avg_speed_now, self.collisions, self.occupancy)
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

    # FIXME: not used for now, because the training could take longer and the reward function could require adjustements
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
                    flow_generation_fix_num_veh(self.model, self.model_idx, self.gen_car_distrib[1], episode_length, self.is_learning)
                elif self.gen_car_distrib[0] == 'bimodal':
                    flow_generation(self.model, self.model_idx, bimodal_distribution_24h(self.gen_car_distrib[1]), 1)
                # self.gen_car_distrib[1] += 50
                sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', sumoExecutable)
                self.sumo_process = subprocess.Popen([sumoBinary, "-c", f"./traffic_environment/sumo/3_2_merge_{self.model}_{self.model_idx}.sumocfg", '--start'] 
                                                     + ["--default.emergencydecel=7"]
                                                    #  + ["--emergencydecel.warning-threshold=1"]
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

        is_sumo_running = psutil.pid_exists(self.sumo_process.pid) if self.sumo_process else False
        if not is_sumo_running:
            self.start_sumo()
        else:
            for segment in [seg_1_before]:
                [traci.lane.setMaxSpeed(segId, self.speed_limit / 3.6) for segId in segment] # km/h to m/s
            # [traci.edge.setFriction(edgeId, self.frictionValue) for edgeId in edges]

        if self.model in cont_act_space_models:
            # Map continuous action to nearest discrete speed limit
            self.speed_limit = self.speed_limits[np.argmin(np.abs(self.speed_limits - action[0]))]
        else:
            # Apply action
            self.speed_limit = self.speed_limits[action]

        veh_time_sum = 0
        num_vehicles_in_merging_zone_temp = 0
        mean_speeds_seg_1_before_mps_temp = 0
        mean_speeds_seg_0_after_mps_temp = 0
        emissions_seg_1_before_temp = 0
        # Step simulation and collect data
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
            if not self.is_learning and not self.test_with_electric:
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
            if not self.is_learning and not self.test_with_disobedient:
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
        self.total_emissions_now = np.sum(emissions_seg_1_before_temp) / self.aggregation_time

        # Update previous state variables
        self.emissions_over_time.append(self.total_emissions_now)
        self.mean_speed_over_time.append(avg_speed_before_1_now * 3.6)  # m/s to km/h

        self.n_before0 = np.average(n_before0_avg) if len(n_before0_avg) > 0 else 0
        self.n_before1 = np.average(n_before1_avg) if len(n_before1_avg) > 0 else 0
        self.n_after = np.average(n_after_avg) if len(n_after_avg) > 0 else 0
        
        # Calculate reward
        self.reward = self.reward_func_wrap()
        
        self.rewards.append(self.reward)

        flow = (veh_time_sum / self.aggregation_time) * (get_full_week_car_generation_rates() * 60 * 60)
        self.flows.append(flow)
        self.density.append(num_vehicles_in_merging_zone_temp / traci.lane.getLength("seg_0_before_0") / 3)

        self.occupancy = self.occupancy_sum / self.aggregation_time

        self.obs = np.array([avg_speed_before_1_now])

        # self.sim_length -= 1  # Reduce simulation length by 1 second
        # done = False
        # if self.sim_length <= 0:
            # if not self.is_learning and traci.edge.getLastStepVehicleNumber("seg_0_before") > 0:
                # pass # wait until no vehicle is present in the merging zone
            # else:
                # done = True
        
        # If there is a previous action, compare it to the current action
        if self.prev_act_spdlim is not None:
            # Calculate the change in action
            action_change = abs(self.speed_limit - self.prev_act_spdlim)
            if action_change != 0:
                self.act_spdlim_change_amounts.append(action_change)
        else:
            # First step, no previous action to compare
            pass
        # Update the previous action
        self.prev_act_spdlim = self.speed_limit

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

        return self.obs, self.reward, (traci.simulation.getMinExpectedNumber() <= 0), False, {}

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
        self.flow = 0
        self.flows = []
        self.total_waiting_time = 0
        # self.sumo_process = None
        self.sumo_max_retries = 5
        self.occupancy_sum = 0
        self.occupancy = 0
        self.rewards = []
        self.avg_speed_now = 0
        self.total_emissions_now = 0
        self.emissions_over_time = []
        self.mean_speed_over_time = []
        self.frictionValue = 1.0 # Dry road: friction = 1.0; Wet road: friction = 0.7; Icy road: friction = 0.3
        self.is_first_step_delay_on = True
        self.density = []
        self.n_before0 = 0
        self.n_before1 = 0
        self.n_after = 0
        self.collisions = 0
        # self.sim_length = episode_length * num_of_hr_intervals * num_of_iterations * num_envs_per_model
        self.speed_limit = 130
        self.obs = np.array([0], dtype=np.float64)
        self.prev_act_spdlim = None
        self.act_spdlim_change_amounts = []
        self.logger = TrafficDataLogger()
        
        return self.obs, {}
    
    def close(self):
        self.close_sumo("close() method called")

    def __del__(self):
        try:
            self.close()
        except AttributeError:
            # Handle cases where attributes might not be initialized
            print("Warning: Attempted to delete an uninitialized TrafficEnv instance.")

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

class RuleBasedTrafficEnv(TrafficEnv):
    def __init__(self, port, model, model_idx, is_learning, base_gen_car_distrib):
        # Pass all required arguments to parent class
        super().__init__(
            port=port,
            model=model,
            model_idx=model_idx,
            is_learning=is_learning,
            base_gen_car_distrib=base_gen_car_distrib,
        )
        self.speed_history = collections.deque(maxlen=15)  # Store last 15 speed measurements
        self.flow_downstream = 0
        self.flow_upstream = 0

    def determine_target_speed(self, current_speed, occupancy, avg_speed_trend, moving_avg_speed):
        """
        Advanced rule-based VSL control considering multiple traffic parameters
        """
        # Flow balance factor (closer to 1 means better balance)
        flow_ratio = self.flow_downstream / self.flow_upstream if self.flow_upstream > 0 else 1

        # Speed trend factor (positive means accelerating)
        speed_trend = avg_speed_trend[-1] - avg_speed_trend[0]
        
        if occupancy > 80:
            base_speed = max(moving_avg_speed - 20, 50)  # Severe congestion: reduce speeds gradually
        elif 40 < occupancy <= 80:
            if flow_ratio < 0.7 and speed_trend < 0:
                base_speed = min(moving_avg_speed + 10, 100)  # Increase speed to clear congestion
            else:
                base_speed = max(moving_avg_speed - 10, 70)  # Gradually decrease speed
        elif 12 < occupancy <= 40:
            if flow_ratio > 0.9:
                base_speed = min(moving_avg_speed + 10, 120)  # Good flow; increase speed
            else:
                base_speed = moving_avg_speed
        else: 
            # Free-flow conditions: prioritize safety while allowing high speeds
            base_speed = min(max(moving_avg_speed + 5, current_speed), 130)

        # Round to nearest available discrete speed limit
        return round(base_speed / 5) * 5
    
    def feedback_adjustment(self):
        # Calculate speed variance as a measure of traffic stability
        speed_variance = np.var(list(self.speed_history))

        # Adjust based on flow ratios and stability
        if self.flow_downstream < self.flow_upstream * 0.8 or speed_variance > 5:
            # Bottleneck detected or unstable traffic: reduce speed
            self.speed_limit = max(self.speed_limit - 10, 50)
        elif self.flow_downstream > self.flow_upstream * 1.2 and speed_variance < 2:
            # Good clearing downstream and stable traffic: increase speed
            self.speed_limit = min(self.speed_limit + 10, 130)

    def step(self, _):
        # Check if SUMO is running
        is_sumo_running = psutil.pid_exists(self.sumo_process.pid) if self.sumo_process else False
        if not is_sumo_running:
            self.start_sumo()
        
        # Determine speed limit using rule-based logic
        current_speed = self.obs[0] * 3.6
                
        # Initialize measurement variables
        veh_time_sum = 0
        mean_speeds_seg_1_before_mps_temp = 0
        mean_speeds_seg_0_after_mps_temp = 0
        emissions_seg_1_before_temp = 0
        self.occupancy_sum = 0

        flow_upstream_temp = 0
        flow_downstream_temp = 0
        
        # Simulation steps and data collection
        for _ in range(self.aggregation_time):
            try:
                traci.simulationStep()
            except (FatalTraCIError, TraCIException):
                logging.debug("Lost connection to SUMO from steps")
                self.close_sumo("lost comm with SUMO")
                return self.reset()

            # Collect measurements
            veh_time_sum += sum([traci.inductionloop.getLastStepVehicleNumber(loop) for loop in loops_after])
            mean_speeds_seg_1_before_mps_temp += traci.edge.getLastStepMeanSpeed("seg_1_before")
            mean_speeds_seg_0_after_mps_temp += traci.edge.getLastStepMeanSpeed("seg_0_after")
            emissions_seg_1_before_temp += traci.edge.getCO2Emission("seg_1_before")
            flow_upstream_temp += traci.edge.getLastStepVehicleNumber("seg_1_before")
            flow_downstream_temp += traci.edge.getLastStepVehicleNumber("seg_0_after")
            
            # Calculate occupancy
            occ_max = 0
            for loops in loops_before:
                occ_loop = sum([traci.inductionloop.getLastStepOccupancy(loop) for loop in loops]) / len(loops)
                if occ_loop > occ_max:
                    occ_max = occ_loop
            self.occupancy_sum += occ_max
        
        # Calculate averages
        self.occupancy = self.occupancy_sum / self.aggregation_time
        avg_speed_before_1_now = np.mean(mean_speeds_seg_1_before_mps_temp) / self.aggregation_time
        
        self.flow_upstream = (flow_upstream_temp * 3600) / self.aggregation_time
        self.flow_downstream = (flow_downstream_temp * 3600) / self.aggregation_time
        
        # Update speed trend history
        self.speed_history.append(avg_speed_before_1_now)
        avg_speed_trend = list(self.speed_history) if len(self.speed_history) == 15 else [avg_speed_before_1_now] * 15

        # Update observation
        self.obs = np.array([avg_speed_before_1_now])

        moving_avg_speed = np.mean(list(self.speed_history))
        
        target_speed = self.determine_target_speed(current_speed, self.occupancy, avg_speed_trend, moving_avg_speed)

        self.speed_limit = round((self.speed_limit + target_speed) / 2 / 5) * 5

        # TBC
        
        # Apply feedback adjustment based on downstream impact
        self.feedback_adjustment()

        # Apply the new speed limit to all lanes
        for segment in [seg_1_before]:
            [traci.lane.setMaxSpeed(segId, self.speed_limit / 3.6) for segId in segment]
        
        # Update observation
        self.obs = np.array([avg_speed_before_1_now])

        # Log metrics
        self.logger.log_step(
            vss=self.speed_limit,
            occupancy=self.occupancy,
            speed_before=avg_speed_before_1_now,
            speed_after=mean_speeds_seg_0_after_mps_temp / self.aggregation_time,
            reward=0, # No reward needed for rule-based control
            flow_downstream=self.flow_upstream,
            flow_upstream=self.flow_downstream,
            avg_speed_trend=avg_speed_trend
        )
        
        return self.obs, 0, (traci.simulation.getMinExpectedNumber() <= 0), False, {}

""" 
Note: DEPRECATED, no intention to be used since manual trial-and-error tuning of gains must be done
"""
# class PIDBasedTrafficEnv(TrafficEnv):
#     def __init__(self, port, model, model_idx, is_learning, base_gen_car_distrib):
#         super().__init__(
#             port=port,
#             model=model,
#             model_idx=model_idx,
#             is_learning=is_learning,
#             base_gen_car_distrib=base_gen_car_distrib
#         )
#         # PID controller parameters
#         self.Kp = 0.5  # Proportional gain
#         self.Ki = 0.1  # Integral gain
#         self.Kd = 0.2  # Derivative gain
#         self.prev_error = 0
#         self.integral = 0
#         self.target_occupancy = 12  # Optimal occupancy target
#         self.dt = 60    # Time step (60 seconds)
#         # Optimization parameters
#         self.optimization_window = 60  # Number of steps to evaluate performance
#         self.performance_history = []
#         self.current_episode_errors = []
#         self.simulation_step = 0
        
#     def pid_speed_control(self, current_occupancy):
#         """PID controller with optimized parameters"""
#         error = self.target_occupancy - current_occupancy
        
#         # Anti-windup for integral term
#         if abs(self.integral) < 100:  # Prevent integral windup
#             self.integral += error * self.dt
        
#         derivative = (error - self.prev_error) / self.dt
        
#         # Calculate control output with optimized parameters
#         output = (self.Kp * error + 
#                  self.Ki * self.integral + 
#                  self.Kd * derivative)
        
#         self.prev_error = error
        
#         # Convert PID output to speed adjustment
#         base_speed = 100
#         speed_adjustment = output * 10
#         target_speed = base_speed + speed_adjustment
        
#         # Log current error for optimization
#         self.current_episode_errors.append(abs(error))
        
#         return min(max(round(target_speed/5)*5, 50), 130)

#     def optimize_pid_parameters(self, num_episodes=5):
#         """Optimize PID parameters using scipy.optimize"""
#         def objective_function(params):
#             self.Kp, self.Ki, self.Kd = params
#             total_error = 0
            
#             # Run multiple episodes to ensure robust parameters
#             for _ in range(num_episodes):
#                 episode_error = self.run_evaluation_episode()
#                 total_error += episode_error
            
#             avg_error = total_error / num_episodes
#             self.optimization_history.append({
#                 'params': params,
#                 'error': avg_error
#             })
#             return avg_error
        
#         # Initial parameters
#         x0 = [self.Kp, self.Ki, self.Kd]
        
#         # Parameter bounds
#         bounds = ((0.1, 2.0),  # Kp bounds
#                  (0.0, 0.5),   # Ki bounds
#                  (0.0, 0.5))   # Kd bounds
        
#         # Optimization constraints
#         constraints = (
#             {'type': 'ineq', 'fun': lambda x: x[0] - x[1]},  # Kp > Ki
#             {'type': 'ineq', 'fun': lambda x: x[0] - x[2]}   # Kp > Kd
#         )
        
#         # Run optimization
#         result = minimize(
#             objective_function,
#             x0,
#             method='SLSQP',
#             bounds=bounds,
#             constraints=constraints,
#             options={'maxiter': 50}
#         )
        
#         # Apply optimized parameters
#         self.Kp, self.Ki, self.Kd = result.x

#         print(f"Optimized PID parameters:")
#         print(f"Kp: {self.Kp:.3f}")
#         print(f"Ki: {self.Ki:.3f}")
#         print(f"Kd: {self.Kd:.3f}")
#         print(f"Final optimization error: {result.fun:.3f}")

#         return result
    
#     def run_evaluation_episode(self):
#         """Run one episode and return the performance metric"""
#         self.reset()
#         episode_errors = []
#         done = False
        
#         while not done:
#             _, _, done, _, _ = self.step(None)
#             error = abs(self.target_occupancy - self.occupancy)
#             episode_errors.append(error)
            
#         # Calculate episode performance metric
#         # Using mean absolute error and penalizing oscillations
#         mean_error = np.mean(episode_errors)
#         error_variance = np.var(episode_errors)
#         performance_metric = mean_error + 0.5 * error_variance
        
#         return performance_metric

#     def step(self, _):
#         is_sumo_running = psutil.pid_exists(self.sumo_process.pid) if self.sumo_process else False
#         if not is_sumo_running:
#             self.start_sumo()
        
#         # Get current traffic state
#         current_speed = self.obs[0] * 3.6
        
#         # Determine speed limit using PID control
#         if self.occupancy > 80:
#             target_speed = 50  # Emergency low speed for severe congestion
#         else:
#             target_speed = self.pid_speed_control(self.occupancy)
        
#         # Find closest available speed limit
#         self.speed_limit = self.speed_limits[np.argmin(np.abs(self.speed_limits - target_speed))]
        
#         # Apply speed limit gradually to create smooth transition
#         speed_difference = abs(self.speed_limit - current_speed)
#         if speed_difference > 20:  # Maximum 20 km/h change per step
#             direction = 1 if self.speed_limit > current_speed else -1
#             self.speed_limit = current_speed + (direction * 20)
#             self.speed_limit = self.speed_limits[np.argmin(np.abs(self.speed_limits - self.speed_limit))]
        
#         # Apply speed limit to all lanes
#         for segment in [seg_1_before]:
#             [traci.lane.setMaxSpeed(segId, self.speed_limit / 3.6) for segId in segment]
        
#         # Initialize measurement variables
#         veh_time_sum = 0
#         mean_speeds_seg_1_before_mps_temp = 0
#         mean_speeds_seg_0_after_mps_temp = 0
#         emissions_seg_1_before_temp = 0
#         self.occupancy_sum = 0
        
#         # Simulation steps and data collection
#         for _ in range(self.aggregation_time):
#             try:
#                 traci.simulationStep()
#             except (FatalTraCIError, TraCIException):
#                 logging.debug("Lost connection to SUMO from steps")
#                 self.close_sumo("lost comm with SUMO")
#                 return self.reset()

#             # Collect measurements
#             veh_time_sum += sum([traci.inductionloop.getLastStepVehicleNumber(loop) for loop in loops_after])
#             mean_speeds_seg_1_before_mps_temp += traci.edge.getLastStepMeanSpeed("seg_1_before")
#             mean_speeds_seg_0_after_mps_temp += traci.edge.getLastStepMeanSpeed("seg_0_after")
#             emissions_seg_1_before_temp += traci.edge.getCO2Emission("seg_1_before")
            
#             # Calculate occupancy
#             occ_max = 0
#             for loops in loops_before:
#                 occ_loop = sum([traci.inductionloop.getLastStepOccupancy(loop) for loop in loops]) / len(loops)
#                 if occ_loop > occ_max:
#                     occ_max = occ_loop
#             self.occupancy_sum += occ_max
        
#         # Calculate averages
#         self.occupancy = self.occupancy_sum / self.aggregation_time
#         avg_speed_before_1_now = np.mean(mean_speeds_seg_1_before_mps_temp) / self.aggregation_time
        
#         # Update observation
#         self.obs = np.array([avg_speed_before_1_now])
        
#         # Log metrics
#         self.logger.log_step(
#             vss=self.speed_limit,
#             occupancy=self.occupancy,
#             speed_before=avg_speed_before_1_now,
#             speed_after=mean_speeds_seg_0_after_mps_temp / self.aggregation_time,
#             reward=0  # No reward needed for rule-based control
#         )
        
#         # Periodically optimize PID parameters
#         if self.simulation_step % 10 == 0:
#             self.optimize_pid_parameters()
#         self.simulation_step += 1

#         return self.obs, 0, (traci.simulation.getMinExpectedNumber() <= 0), False, {}
