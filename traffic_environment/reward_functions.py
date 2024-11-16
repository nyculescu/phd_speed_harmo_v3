import math
import numpy as np
import unittest
from parameterized import parameterized
import traci
from traffic_environment.setup import create_sumocfg
import os
from config import sumoExecutable, car_following_model
import subprocess
from traffic_environment.flow_gen import flow_generation_wrapper
import logging
import csv
from traffic_environment.road import seg_0_before, seg_1_before
import xml.etree.ElementTree as ET
from traci.exceptions import FatalTraCIError, TraCIException
from datetime import datetime
from glob import glob
import pandas as pd

max_emissions = 250000 # empirical data after running the simulation at max capacity
max_occupancy = 50000 # empirical data after running the simulation at max capacity

# Logistic function parameters
num_halts_k = 0.1  # steepness of the curve
num_halts_x_0 = 40  # midpoint of number of halts where reward is 0.5

def normalize(value, min_value, max_value):
    """Normalize a value between min_value and max_value to a range [0, 1]."""
    norm_val = (value - min_value) / (max_value - min_value) if (min_value <= value <= max_value) else 1
    return norm_val

def quad_occ_reward(occupancy):
    ''' This function rewards occupancy rates around 12% the most, with decreasing rewards for both lower and higher occupancies.
        - low occupancy rates (0-12%) => increases from 0.5 at x=0 to 1 at x=12
        - medium to high occupancy rates (12-80%) => decreases from 1 at x=12 to 0 at x=80
        - very low (≤0) or very high (≥80) occupancy rates => reward is 0
        The function is continuous at x=12, where both pieces evaluate to 1. However, it is not differentiable at x=12 due to the change in slope.
        Ref.: Kidando, E., Moses, R., Ozguven, E. E., & Sando, T. (2017). Evaluating traffic congestion using the traffic occupancy and speed distribution relationship: An application of Bayesian Dirichlet process mixtures of generalized linear model. Journal of Transportation Technologies, 7(03), 318.
    '''
    if 0 < occupancy <= 12:
        return ((0.5 * occupancy) + 6) / 12
    elif 12 < occupancy < 80:
        return ((occupancy-80)**2/68**2)
    else:
        return 0

# Minimize emissions by providing higher rewards for lower emission levels.
def emission_reward(avg_emission, penalty=-0.5):
    """
    Calculate the reward based on CO2 emissions.

    Parameters:
    - avg_emission: float, total CO2 emissions at the time of measurement.
    - max_emissions: float, maximum acceptable CO2 emissions for normalization.
    - penalty: float, penalty value for emissions exceeding max_emissions.

    Returns:
    - reward: float, normalized reward between 0 and 1, or a penalty if emissions are too high.
    """
    if avg_emission > max(0, max_emissions):
        return penalty
    # Calculate the reward
    reward = max(0, 1 - (avg_emission / max_emissions))
    
    return reward

def enhanced_co2_emission_reward(CO2_emissions, dynamic_max_CO2, delta_CO2_emissions, delta_t, alpha=0.1):
    '''
    a more nuanced reward function could consider not just the absolute level of emissions but also the rate of change in emissions over time or relative to traffic flow
    Dynamic Scaling: Adjusting the maximum CO2 value dynamically based on real-time traffic conditions or historical data.
    Rate of Change: Incorporating a penalty for increasing emission trends over time.
    reward = max (0, 1 - CO2 emissions / dynamic max CO2)-alpha * delta CO2 emissions / delta t)
    - dynamic_max_CO2: Adjusted maximum CO2 emissions based on current conditions.
    - α: A weight factor for the rate of change penalty.
    - ΔCO2 emissions: Change in emissions over a time interval Δt.
    '''
    if dynamic_max_CO2 <= 0:
        raise ValueError("dynamic_max_CO2 must be greater than zero.")
    
    base_reward = max(0, 1 - (CO2_emissions / dynamic_max_CO2))
    change_penalty = alpha * (delta_CO2_emissions / delta_t)
    reward = max(0, base_reward - change_penalty)
    
    return reward

def enhanced_co2_emission_reward_2(delta_co2, avg_speed, optimal_speed=88/3.6, alpha=1.0, beta=1.0):
    """
    Calculate the reward based on changes in CO2 emissions and average speed.

    Parameters:
    - optimal_speed: float, the speed (in m/s) that minimizes CO2 emissions based on 
        1. Gao, C., Xu, J., Jia, M., & Sun, Z. (2024). Correlation between carbon emissions, fuel consumption of vehicles and speed limit on expressway. Journal of Traffic and Transportation Engineering (English Edition).
        2. Haworth, N., & Symmons, M. (2001). Driving to reduce fuel consumption and improve road safety. Monash University Accident Research Centre, 34-38.
    - delta_co2: float, change in CO2 emissions between time steps.
    - alpha: float, weight for CO2 reduction importance.
    - beta: float, weight for speed increase importance.

    Returns:
    - reward: float, calculated reward value.
    """
    
    # Calculate reward based on CO2 emissions and average speed
    # Assuming higher speeds are better (less congestion)
    reward = -alpha * delta_co2 + beta * avg_speed

    return reward

# # Minimize total travel time by rewarding faster completion of trips.
# def travel_time_reward(total_travel_time, max_travel_time=3600):
#     # Normalize travel time to a range [0, 1]
#     normalized_travel_time = min(total_travel_time / max_travel_time, 1)
#     return 1 - normalized_travel_time  # Higher reward for shorter travel times

'''
    Dynamic Normalization: Offers flexibility and can be tailored to specific traffic conditions but requires accurate real-time data or robust predictive models.
    Logistic Function: Provides a consistent and smooth reward structure that naturally accommodates exponential growth in halts without needing dynamic inputs.
'''

def dynamic_normalization_reward(num_halts, baseline_halts, vehicle_count, congestion_factor=0.5):
    """
    Parameters:
        - vehicle_count: int, current number of vehicles.
        - baseline_halts: int, baseline number of halts under normal conditions.
        - congestion_factor: float, factor to scale halts with increasing vehicles.
    """
    dynamic_max_halts = max(0, baseline_halts + int(congestion_factor * vehicle_count))
    if dynamic_max_halts <= 0:
        raise ValueError("dynamic_max_halts must be greater than zero.")
    reward = 1 - (num_halts / dynamic_max_halts)
    return max(0, min(1, reward))

def logistic_halts_reward(num_halts, x0=5, k=1):
    """
    Calculate the reward based on the number of halts using a logistic function.

    Parameters:
    - num_halts: int, number of stops.
    - x0: float, midpoint where reward transitions.
    - k: float, steepness of the transition curve.

    Returns:
    - reward: float, normalized reward between 0 and 1.
    """
    # Calculate logistic reward
    reward = 1 / (1 + math.exp(k * (num_halts - x0)))
    
    return reward

def basic_num_halts_reward(num_halts, max_num_halts=10):
    """
    Calculate the reward based on the number of halts.
    
    :param number_of_halts: The number of times vehicles come to a complete stop.
    :param max_halts: The maximum number of halts for full decay.
    :return: Calculated reward.
    """
    if num_halts < 0:
        raise ValueError("Number of halts cannot be negative")
    
    # Calculate reward
    reward = max(0, 1 - num_halts / max_num_halts)
    return reward

# Encourage speeds close to a target speed (e.g., optimal flow speed) while penalizing speeds that are too low or too high.
# penalty-based system where deviations from the target speed reduce the reward.
def speed_reward(mean_speed_mps, target_speed_mps):
    # Reward is maximum when mean speed is at target speed
    if mean_speed_mps < 0:
        return 0
    elif mean_speed_mps <= target_speed_mps:
        return mean_speed_mps / target_speed_mps
    else:
        # Penalize speeds significantly higher than the target to ensure safety
        return max(0, 1 - ((mean_speed_mps - target_speed_mps) / target_speed_mps))

# def wait_time_penalty(total_waiting_time):
#     # Penalize both total waiting time and number of stops
#     return (-1) * (total_waiting_time / 3600)

def complex_reward(target_speed_mps, mean_speeds_mps, occupancy, mean_emissions, mean_num_halts):
    weight_speed = 0.25
    weight_occupancy = 0.3
    weight_emissions = 0.35
    weight_num_halts = 0.1

    # Calculate rewards for each parameter
    spd_reward = speed_reward(mean_speeds_mps, target_speed_mps)
    emis_reward = emission_reward(mean_emissions)
    occ_reward = quad_occ_reward(occupancy)
    halts_reward = basic_num_halts_reward(mean_num_halts)

    # Combine individual rewards into a total reward
    reward = (
        occ_reward * weight_occupancy +
        spd_reward * weight_speed +
        emis_reward * weight_emissions +
        halts_reward * weight_num_halts
    )

    return reward

"""
    Emission Reduction: Increase the weight of emission reduction if it's a priority. For example, try k1 = -0.5 and b1 = 1.0 to emphasize reducing emissions more heavily.
    Flow Efficiency: Adjust to prioritize speed consistency. Try k3 = -0.3 and b3 = 1.0 to focus more on maintaining speed.
"""
def reward_co2_avgspeed(prev_emissions, total_emissions_now, prev_mean_speed, avg_speed_now, k1=-0.4, b1=0.95, k3=-0.25, b3=0.95):
    # Normalize emissions
    normalized_emissions_t = total_emissions_now / 30000000  # Example normalization factor
    normalized_emissions_t_minus_1 = np.sum(prev_emissions) / 30000000

    # Calculate emission reduction component
    emission_reduction = k1 * (normalized_emissions_t - b1 * normalized_emissions_t_minus_1)
    
    # Calculate traffic flow efficiency component
    flow_efficiency = k3 * (avg_speed_now - b3 * np.mean(prev_mean_speed))
    
    # Total reward
    reward = emission_reduction + flow_efficiency
    
    # Clip reward to prevent extreme values
    reward = max(min(reward, 100), -100)
    
    return reward

def reward_function(flow_merge, density_merge, collision_occurred, overtaking_occurred):
    # Constants
    epsilon_r = 0.01
    w_f = 1      # Weight for flow
    w_d = 0.5    # Weight for density
    r_c = -100   # Penalty for collision (strongly discourage unsafe driving behavior)
    r_o = 10     # Reward for safe overtaking
    
    if collision_occurred:
        return epsilon_r / (epsilon_r + w_f * flow_merge + w_d * density_merge) + r_c
    elif overtaking_occurred:
        return epsilon_r / (epsilon_r + w_f * flow_merge + w_d * density_merge) + r_o
    else:
        return epsilon_r / (epsilon_r + w_f * flow_merge + w_d * density_merge)

def reward_function_v2(n_before, n_after, emissions, avg_speed, collision_occurred=2, aggregation_time=60):
    """
        Get vehicle numbers before and after merging point; emissions and avg speed before merging point
        Explanation of Parameters:
            Q_{\text{out}}: The outflow of vehicles after merging, calculated as traci.edge.getLastStepVehicleNumber("seg_0_after") / aggregation_time.
            n_{\text{before}}: Vehicle accumulation before merging, calculated as traci.edge.getLastStepVehicleNumber("seg_0_before").
            n_{\text{opt}}: Optimal vehicle accumulation where traffic flow is maximized. This value needs to be tuned based on experiments or real-world data.
            penalty\_accumulation : A Gaussian penalty that increases as vehicle accumulation deviates from the optimal value (n_{\text{opt}}).
            emissions: CO2 emissions from vehicles in the section before merging, calculated using traci.edge.getCO2Emission("seg_0_before").
            avg\_speed : Average speed of vehicles before merging, calculated using traci.edge.getLastStepMeanSpeed("seg_0_before").
            target\_speed : Target speed for smooth traffic flow, set to a reasonable value like 25 km/h converted to m/s.
            Weights (w_e,w_s): These weights control how much emphasis is placed on reducing emissions and maintaining target speeds relative to maximizing flow.
    """
    # Outflow calculation (vehicles passing through after merging)
    Q_out = n_after / aggregation_time
    
    # Define optimal accumulation and variance for penalty
    n_opt = 50  # Optimal vehicle accumulation
    sigma = 10  # Standard deviation for Gaussian penalty
    
    r_c = 0 if collision_occurred <= 2 else -1  # Penalty for collision

    # Calculate Gaussian penalty based on deviation from optimal accumulation
    penalty_accumulation = np.exp(-((n_before - n_opt) ** 2) / (2 * sigma ** 2)) + r_c
    
    target_speed = 120 / 3.6  # Target speed in m/s

    w_e = 0.00 # Weight for emissions
    w_s = 0.00 # Weight for speed
    
    # Reward calculation
    reward = Q_out * penalty_accumulation - w_e * emissions - w_s * max(target_speed - avg_speed, 0) ** 2
    
    return reward

def reward_function_v3(model, n_before, n_after, n_lanes = 3, seg0_before_merge_length = 275 + 300, seg0_after_merge_length = 500, collision_occurred=2, aggr_time=60):
    """
        Get vehicle numbers before and after merging point; emissions and avg speed before merging point
        Explanation of Parameters:
            Q_{\text{out}}: The outflow of vehicles after merging, calculated as traci.edge.getLastStepVehicleNumber("seg_0_after") / aggregation_time.
            n_{\text{before}}: Vehicle accumulation before merging, calculated as traci.edge.getLastStepVehicleNumber("seg_0_before").
            n_{\text{opt}}: Optimal vehicle accumulation where traffic flow is maximized. This value needs to be tuned based on experiments or real-world data.
            penalty\_accumulation : A Gaussian penalty that increases as vehicle accumulation deviates from the optimal value (n_{\text{opt}}).
    """
    epsilon_r = 0.01

    c_f = n_before / aggr_time # Flow rate before merging
    w_f = 0.5 # Weight for flow

    c_d = n_before / (n_lanes * seg0_before_merge_length) # Density before merging
    w_d = 0.5 # Weight for density
    
    r_c = -10 if collision_occurred <= 2 else -1  # Penalty for collision

    # Reward calculation
    reward = (epsilon_r / (epsilon_r + w_f * c_f + w_d * c_d)) + r_c
    
    return reward

def reward_function_v4(model, n_before, n_after, avg_speed, collisions, 
                       n_lanes=3, seg01_before_merge_length=275 + 300, seg0_after_merge_length=500,
                       collision_occurred=2, aggr_time=60):
    """
    Reward function for traffic management in a SUMO simulation using deep reinforcement learning.
    
    Parameters:
    - n_before: Number of vehicles before the merging point.
    - n_after: Number of vehicles after the merging point.
    - avg_speed: List of average speeds of vehicles before the merging point (in km/h).
    - collisions: Number of collisions detected in the simulation step.
    - n_lanes: Number of lanes before merging.
    - seg0_before_merge_length: Length of segment before merging (in meters).
    - seg0_after_merge_length: Length of segment after merging (in meters).
    - collision_occurred: Threshold for penalizing collisions.
    - aggr_time: Aggregation time window for calculating flow rate and density (in seconds).
    
    Returns:
    - reward: Calculated reward based on traffic flow, density, speed, and safety considerations.
    """
    
    # Constants
    epsilon_r = 0.01  # Small constant to avoid division by zero
    optimal_speed = 60  # Optimal speed in km/h

    # Flow rate calculation
    c_f_before = n_before / aggr_time  # Flow rate before merging
    c_f_after = n_after / aggr_time  # Flow rate after merging
    w_f = 0.4  # Weight for flow

    # Density calculation
    c_d_before = n_before / (n_lanes * seg01_before_merge_length)  # Density before merging
    c_d_after = n_after / (n_lanes * seg0_after_merge_length)  # Density after merging
    w_d = 0.3  # Weight for density

    # Collision penalty
    r_c = -100 if collisions >= collision_occurred else 0  # Heavier penalty for more than threshold collisions

    # Speed control reward/penalty
    avg_speed_value = np.mean(avg_speed) if isinstance(avg_speed, list) else avg_speed  # Calculate average if it's a list

    if avg_speed_value < optimal_speed:
        r_s = -(optimal_speed - avg_speed_value) / optimal_speed  # Penalize if below optimal speed
    else:
        r_s = min((avg_speed_value - optimal_speed) / optimal_speed, 1)  # Cap at max reward if above optimal speed
    
    w_s = 0.2  # Weight for speed

    # Convergence term (to encourage stability)
    convergence_factor = abs(c_f_before - c_f_after) + abs(c_d_before - c_d_after)
    
    w_convergence = 0.1  # Weight for convergence factor
    r_convergence = -(convergence_factor)  # Penalize large differences between before and after merging
    
    # Final reward calculation
    reward = (
        epsilon_r / (epsilon_r + w_f * (c_f_before + c_f_after) + w_d * (c_d_before + c_d_after)) +
        w_s * r_s + 
        r_c + 
        w_convergence * r_convergence
    )
    
    return reward


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Unit Testing with Predefined Values """
class TestCalculateReward(unittest.TestCase):
    @parameterized.expand([
        ("no_collision_no_overtaking", 20, 30, False, False, 2.8563267637817762e-05),
        ("collision_occurred", 15, 25, True, False, 100.00003635041803),
        ("overtaking_occurred", 25, 20, False, True, 10.000028563267637),
        ("high_density", 5, 100, False, False, 1.8178512997636793e-05),
        ("zero_flow", 0, 50, False, False, 3.9984006397441024e-05)
    ])
    def test_calculate_reward(self, name: str,
                              flow_merge: int,
                              density_merge: int,
                              collision_occurred: bool,
                              overtaking_occurred: bool,
                              expected: float):
        result = reward_function(flow_merge,
                                  density_merge,
                                  collision_occurred,
                                  overtaking_occurred)
        self.assertAlmostEqual(result, expected, places=5)

def test_and_tune_reward_function():
    """ Simulating Different Traffic Conditions to see how the reward function 
    behaves under different circumstances before running full training sessions """
    flow_merge                  = [10, 20, 35] # Average flow in vehicles per time unit
    density_merge               = [20, 50, 100]   # Average density in vehicles per kilometer
    collision_occurred_cases    = [False, True]
    overtaking_occurred_cases   = [False, True]

    results = []

    # Iterate over scenarios and calculate rewards
    for flow in flow_merge:
        for density in density_merge:
            for collision_occurred in collision_occurred_cases:
                for overtaking_occurred in overtaking_occurred_cases:
                    reward = reward_function(flow, density, collision_occurred, overtaking_occurred)
                    results.append((flow, density, collision_occurred, overtaking_occurred, reward))

    # Display results
    for result in results:
        flow_val = result[0]
        density_val = result[1]
        status = "Collision" if result[2] else "No Collision"
        action = "Overtaking" if result[3] else "No Overtaking"
        print(f"Flow: {flow_val}, Density: {density_val}, {status}, {action}, Reward: {result[4]}")

def reset_additionalfile_for_VSS(speed, t0 = 0, xml_file="traffic_environment/sumo/variable_speed_limits_signs.add.xml"):
    # Define the base content of your additional file (the template)
    base_xml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
    <additional xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
        <variableSpeedSign id="VSS_0" lanes="seg_1_before_0">
            <step time="{t0}" speed="{speed}"/>
        </variableSpeedSign>
        <variableSpeedSign id="VSS_1" lanes="seg_1_before_1">
            <step time="{t0}" speed="{speed}"/>
        </variableSpeedSign>
        <variableSpeedSign id="VSS_2" lanes="seg_1_before_2">
            <step time="{t0}" speed="{speed}"/>
        </variableSpeedSign>
    </additional>'''
    with open(xml_file, 'w') as f:
        f.write(base_xml_content)

def append_additionalfile_VSS(vss_id, time, speed):
    xml_file="traffic_environment/sumo/variable_speed_limits_signs.add.xml"
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Find the variableSpeedSign by ID
    for vss in root.findall('variableSpeedSign'):
        if vss.attrib['id'] == vss_id:
            # Create a new step element
            new_step = ET.Element('step')
            new_step.set('time', str(time))
            new_step.set('speed', str(speed))
            
            # Append the new step to the variableSpeedSign
            vss.append(new_step)
    
    # Write the updated XML back to file
    tree.write(xml_file)

def reward_function_calibration(veh_gen_per_hour, folder_to_store):
    model = "DQN"
    new_speed_limit = 50
    old_speed_limit = new_speed_limit
    peed_limit_run = 4 # hours
    speed_limits = 10
    model_name = 'REW_TST'
    create_sumocfg(model_name, 1)
    # mock_daily_pattern_test = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]
    mock_daily_pattern_test = [veh_gen_per_hour] * peed_limit_run * speed_limits
    # multiply_factor = 15
    # mock_daily_pattern_test = [x * multiply_factor for x in mock_daily_pattern_test]

    # reset_additionalfile_for_VSS(-1) # NOTE: no need to

    flow_generation_wrapper(model_name, 0, 1, mock_daily_pattern_test)

    sumo_port = 9000
    sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', sumoExecutable)
    sumo_process = subprocess.Popen([sumoBinary, "-c", f"./traffic_environment/sumo/3_2_merge_{model_name}_0.sumocfg", '--start'] 
                                         + ["--default.emergencydecel=7"]
                                        #  + ["--emergencydecel.warning-threshold=1"]
                                         + ['--random-depart-offset=3600']
                                         + ["--remote-port", str(sumo_port)] 
                                         + ["--quit-on-end"] 
                                           ,
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)
    traci.init(sumo_port)

    # Init the allowed Max Speed 
    # [[traci.lane.setMaxSpeed(segId, new_speed_limit / 3.6) for segId in edgeId] for edgeId in [seg_1_before]]
    # [[traci.lane.setMaxSpeed(segId, new_speed_limit / 3.6) for segId in edgeId] for edgeId in [seg_0_before]]
    [[traci.lane.setMaxSpeed(segId, new_speed_limit / 3.6) for segId in edgeId] for edgeId in [seg_0_before, seg_1_before]]
    
    n_before_all = 0
    seconds_passed = 0
    seconds_passed_ = 0
    aggregation_interval = 60 # seconds
    csv_dir = os.path.join(os.getcwd(), f'eval/rew_func/{folder_to_store}')
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    csv_path = os.path.join(csv_dir, f"RewFuncCalib_VehpHr_{veh_gen_per_hour}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    emissions = []
    avg_speed = []
    vehicles_on_road = []
    n_before_avg = []
    n_after_avg = []
    n_before_all_avg = []
    collisions = 0
    
    with open(csv_path, mode='w', newline='') as csvfile:
        fieldnames = ['Minute', 'Reward', 'Vehicles before', 'Vehicles after', 'Emissions', 'Avg Speed', 
                      'Vehicles on road', 'VSS', "Vehicles generated each hour", "Collisions"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        while seconds_passed < 60 * aggregation_interval * len(mock_daily_pattern_test):
            try:
                traci.simulationStep()
            except (FatalTraCIError, TraCIException):
                logging.debug("SUMO failed from reward_function_calibration")
            
            # for vehId in traci.vehicle.getIDList(): # NOTE: no need to
            #     traci.vehicle.setSpeedMode(vehId, 31)  # Enforce strict adherence to VSS

            for i in range(10):
                n_before_all += traci.edge.getLastStepVehicleNumber(f"seg_{i+1}_before")
            n_before = traci.edge.getLastStepVehicleNumber("seg_0_before")
            n_after = traci.edge.getLastStepVehicleNumber("seg_0_after") + traci.edge.getLastStepVehicleNumber("seg_1_after")
            n_before_avg.append(n_before)
            n_after_avg.append(n_after)
            n_before_all_avg.append(n_before_all)
            collisions += len(traci.simulation.getCollidingVehiclesIDList())

            if (n_after >= 1 or n_before >= 1) and n_before_all >= 1:
                avg_speed.append(traci.edge.getLastStepMeanSpeed("seg_0_before") * 3.6)
                emissions.append(traci.edge.getCO2Emission("seg_0_before"))
                vehicles_on_road.append(len(traci.vehicle.getLoadedIDList()))

            # Adjust vehicle generation rate based on traffic volume
            if seconds_passed % aggregation_interval == 0: # each minute
                avg_speed_temp = np.average(avg_speed) if len(avg_speed) > 0 else 0
                emissions_temp = np.average(emissions) if len(emissions) > 0 else 0
                vehicles_on_road_temp = np.average(vehicles_on_road) if len(vehicles_on_road) > 0 else 0
                n_before_temp = np.average(n_before_avg) if len(n_before_avg) > 0 else 0
                n_after_temp = np.average(n_after_avg) if len(n_after_avg) > 0 else 0
                n_before_all_temp = np.average(n_before_all_avg) if len(n_before_all_avg) > 0 else 0
                reward = reward_function_v4(model=None, n_before=n_before, n_after=n_after,
                            avg_speed=avg_speed, collisions=collisions)
                
                if (n_after >= 1 or n_before_temp >= 1) and n_before_all_temp >= 1:
                    logging.debug(f"Minute {seconds_passed / aggregation_interval}; "
                        f"Reward: {reward}; "
                        f"Vehicles before: {n_before_temp:.2f}; "
                        f"Vehicles after: {n_after_temp:.2f}; "
                        f"Emissions: {emissions_temp:.2f}; "
                        f"Avg Speed: {avg_speed_temp:.2f}; "
                        f"Vehicles on road: {vehicles_on_road_temp:.2f}; ",
                        # f"VSS: {new_speed_limit:.2f}"
                        )
                    writer.writerow({
                        'Minute': seconds_passed / aggregation_interval,
                        'Reward': reward,
                        'Vehicles before': n_before_temp,
                        'Vehicles after': n_after_temp,
                        'Emissions': emissions_temp,
                        'Avg Speed': avg_speed_temp,
                        'Vehicles on road': vehicles_on_road_temp,
                        'VSS': new_speed_limit,
                        'Vehicles generated each hour': veh_gen_per_hour,
                        'Collisions': collisions
                    })
                    # TODO: update also fieldnames in case of adding/removing a field from the csv

                    avg_speed.clear()
                    emissions.clear()
                    vehicles_on_road.clear()
                    n_before_avg.clear()
                    n_after_avg.clear()
                    n_before_all_avg.clear()

                    if (seconds_passed % (3600 * peed_limit_run) <= (seconds_passed_ / 3)) and seconds_passed_ > (3600 * peed_limit_run):
                        new_speed_limit += 10
                        seconds_passed_ = 0
                    if new_speed_limit != old_speed_limit:
                        old_speed_limit = new_speed_limit
                        # Variant 1 NOTE: doesn't work. Maybe because it reads the add.xml file at the beginning of the simulation
                        # [append_additionalfile_VSS(f'VSS_{i}', seconds_passed + 60, new_speed_limit / 3.6) for i in range(3)]

                        # Variant 3 NOTE: Works
                        [[traci.lane.setMaxSpeed(segId, new_speed_limit / 3.6) for segId in edgeId] for edgeId in [seg_0_before, seg_1_before]]

                        # Variant 4 NOTE: Doesn't work
                        # [traci.variablespeedsign.setParameter(f"VSS_{i}", "speed", new_speed_limit / 3.6) for i in range(3)] # FIXME: no effect on VSS in SUMO

                    # Variant 2 NOTE: it doesn't work as expected
                    # for vehId in [traci.lane.getLastStepVehicleIDs(laneId) for laneId in [f"seg_1_before_{i}" for i in range(3)]]:
                    #     if vehId and vehId[0]:
                    #         traci.vehicle.setSpeedMode(vehId[0], 31)
                    #         traci.vehicle.slowDown(vehId[0], new_speed_limit / 3.6, 3600)

                elif n_after_temp == 0 and n_before_all_temp == 0:
                    logging.warning("No vehicles detected in the simulation.")
                elif n_after_temp == 0 and n_before == 0 and n_before_all_temp > 0:
                    seconds_passed -= 1
                    seconds_passed_ -= 1
                else:
                    pass
                
            seconds_passed += 1
            seconds_passed_ += 1

    traci.close()    

def merge_dataset(sub_folder):
    # Set the directory path where your CSV files are located
    dir_path = os.path.join(os.getcwd(), f'eval\\rew_func\\{sub_folder}')

    # Collect all CSV files matching the pattern
    csv_files = glob(os.path.join(dir_path, 'RewFuncCalib_VehpHr_*.csv'))

    # Ensure there are CSV files to process
    if not csv_files:
        print("No CSV files found.")
    else:
        # Function to extract vehicle generation rate and timestamp from the filename
        def extract_info(filename):
            base_name = os.path.basename(filename)
            parts = base_name.split('_')
            parts[4], _ = parts[4].split('.')
            
            # Ensure file has at least 5 parts (4 underscores expected)
            if len(parts) < 5:
                raise ValueError(f"Filename does not match expected pattern: {filename}")
            
            try:
                veh_gen_per_hour = int(parts[2])  # Extract vehicle generation rate (e.g., 1000)
                # Remove '.csv' from the last part of the filename (timestamp)
                timestamp_str = parts[3] + "_" + parts[4]  # Extract date and time (e.g., 20241115_170034)
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')  # Convert to datetime object
                return veh_gen_per_hour, timestamp
            except (ValueError, IndexError) as e:
                raise ValueError(f"Error parsing filename: {filename}. Error details: {e}")

        try:
            # Sort files based on vehicle generation rate and timestamp
            csv_files.sort(key=lambda f: extract_info(f))

            # Extract date from the first file for filtering purposes
            first_file_info = extract_info(csv_files[0])
            start_num = first_file_info[0]  # Vehicle generation per hour (e.g., 100)
            date_str = first_file_info[1].strftime('%Y%m%d')  # Date part (e.g., 20241115)

            # Filter files that match the same date and have veh_gen_per_hour <= 3000
            filtered_files = [f for f in csv_files if extract_info(f)[1].strftime('%Y%m%d') == date_str and extract_info(f)[0] <= 3000]

            if not filtered_files:
                print("No relevant CSV files found.")
            else:
                # Merge all filtered CSV files into one DataFrame
                merged_df = pd.concat([pd.read_csv(f) for f in filtered_files])

                # Create the output filename with merged pattern
                veh_gen_per_hour = start_num  # Use the starting vehicle generation rate for naming
                output_filename = f"RewFuncCalib_VehpHr_MERGED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                output_path = os.path.join(dir_path, output_filename)

                # Save the merged DataFrame to a new CSV file
                merged_df.to_csv(output_path, index=False)
                
                print(f"Merged CSV saved as: {output_path}")

        except ValueError as e:
            print(e)

def create_synthetic_data():
    veh_gen_per_hour = 4000
    csv_folder = '2024-11-16_2'

    # while veh_gen_per_hour <= 3500:
    #     reward_function_calibration(veh_gen_per_hour, csv_folder)
    #     veh_gen_per_hour += 50

    reward_function_calibration(veh_gen_per_hour, csv_folder)

    merge_dataset(csv_folder)

# Run the unit tests using TestLoader instead of makeSuite
if __name__ == '__main__':
    # test_and_tune_reward_function()
    # unittest.main()
    create_synthetic_data()
