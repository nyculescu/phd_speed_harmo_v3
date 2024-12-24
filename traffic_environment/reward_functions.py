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
from traffic_environment.road import *
import xml.etree.ElementTree as ET
from traci.exceptions import FatalTraCIError, TraCIException
from datetime import datetime
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import pandas as pd
import json
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.stats.stattools import durbin_watson
import traci.constants as tc

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

max_emissions = 250000 # empirical data after running the simulation at max capacity
max_occupancy = 50000 # empirical data after running the simulation at max capacity

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
                       n_lanes=3, seg0_before_merge_length=275 + 300, seg0_after_merge_length=500,
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
    optimal_speed_min = 90  # Minimum optimal speed in km/h
    optimal_speed_max = 130  # Maximum optimal speed in km/h
    # max_speed = 130  # Maximum allowed speed

    # Flow rate calculation
    c_f_before = n_before / aggr_time  # Flow rate before merging
    c_f_after = n_after / aggr_time  # Flow rate after merging
    w_f = 0.4  # Weight for flow

    # Density calculation
    c_d_before = n_before / (n_lanes * seg0_before_merge_length)  # Density before merging
    c_d_after = n_after / (n_lanes * seg0_after_merge_length)  # Density after merging
    w_d = 0.3  # Weight for density

    # Collision penalty
    r_c = -100 if collisions >= collision_occurred else 0  # Heavier penalty for more than threshold collisions

    # Speed control reward/penalty (penalize large deviations from optimal speed)
    
    avg_speed_value = np.mean(avg_speed) if isinstance(avg_speed, list) else avg_speed  # Calculate average if it's a list
    
    if avg_speed_value < optimal_speed_min:
        r_s = -(optimal_speed_min - avg_speed_value) / optimal_speed_min  # Penalize if below minimum optimal speed
    elif avg_speed_value > optimal_speed_max:
        r_s = -(avg_speed_value - optimal_speed_max) / optimal_speed_max  # Penalize if above maximum allowed speed
    else:
        r_s = min((avg_speed_value - optimal_speed_min) / (optimal_speed_max - optimal_speed_min), 1)  # Reward within range
    
    w_s = 0.2  # Weight for speed

    # Convergence term (to encourage stability)
    convergence_factor = abs(c_f_before - c_f_after) + abs(c_d_before - c_d_after)
    
    w_convergence = 0.1  # Weight for convergence factor
    r_convergence = -(convergence_factor)  # Penalize large differences between before and after merging
    
    # Final reward calculation: Add stronger penalties for density and flow imbalances
    reward = (
        epsilon_r / (epsilon_r + w_f * (c_f_before + c_f_after) + w_d * (c_d_before + c_d_after)) +
        w_s * r_s + 
        r_c + 
        w_convergence * r_convergence
    )
    
    return reward

def reward_function_v5(model, n_before, n_after, avg_speed, collisions,
                       n_lanes=3, seg0_before_merge_length=575,
                       seg0_after_merge_length=500, collision_occurred=4,
                       aggr_time=60):
    
    # Constants
    epsilon_r = 0.01  # Small constant to avoid division by zero
    max_speed = 130  # Maximum allowed speed in km/h

    # Flow rate calculation
    c_f_before = n_before / aggr_time  # Flow rate before merging
    c_f_after = n_after / aggr_time  # Flow rate after merging
    w_f = 0.6  # Weight for flow

    # Density calculation
    c_d_before = n_before / (n_lanes * seg0_before_merge_length)  # Density before merging
    c_d_after = n_after / (n_lanes * seg0_after_merge_length)  # Density after merging
    w_d = 0.4  # Weight for density

    # Collision penalty
    r_c = -100 if collisions >= collision_occurred else 0  # Heavier penalty for more than threshold collisions

    # Dynamic Optimal Speed Calculation (based on density and flow)
    avg_density = (c_d_before + c_d_after) / 2
    avg_flow = (c_f_before + c_f_after) / 2
    
    # Adjust optimal speed based on density and flow
    if avg_density > 0.05:  # High density -> Lower optimal speed
        optimal_speed = max(60, min(90, max_speed - avg_density * 100))
    else:  # Low density -> Higher optimal speed
        optimal_speed = min(130, max(90, max_speed - avg_density * 50))
    
    # Speed control reward/penalty (penalize large deviations from dynamic optimal speed)
    
    avg_speed_value = np.mean(avg_speed) if isinstance(avg_speed, list) else avg_speed
    
    if avg_speed_value < optimal_speed:
        r_s = -(optimal_speed - avg_speed_value) / optimal_speed  # Penalize if below optimal speed
    elif avg_speed_value > max_speed:
        r_s = -(avg_speed_value - max_speed) / max_speed  # Penalize if above maximum allowed speed
    else:
        r_s = min((avg_speed_value - optimal_speed) / optimal_speed, 1)  # Cap at max reward if within range
    
    w_s = 0.3  # Weight for speed

    # Convergence term (to encourage stability)
    convergence_factor = abs(c_f_before - c_f_after) + abs(c_d_before - c_d_after)
    
    w_convergence = 0.15  # Weight for convergence factor
    r_convergence = -(convergence_factor)  # Penalize large differences between before and after merging
    
    # Final reward calculation: Add stronger penalties for density and flow imbalances
    reward = (
        epsilon_r / (epsilon_r + w_f * (c_f_before + c_f_after) + w_d * (c_d_before + c_d_after)) +
        w_s * r_s + 
        r_c + 
        w_convergence * r_convergence
    )
    
    return reward

def reward_function_v6(model, n_before, n_after, avg_speed, collisions,
                       n_lanes=3, seg0_before_merge_length=575,
                       seg0_after_merge_length=500, collision_occurred=5,
                       aggr_time=60):    
    # Constants
    epsilon_r = 0.01  # Small constant to avoid division by zero
    max_speed = 130  # Maximum allowed speed in km/h
    optimal_density = 0.05  # Optimal density (vehicles per meter)
    max_density = 0.1  # Maximum acceptable density before heavy penalties

    # Flow rate calculation
    c_f_before = n_before / aggr_time  # Flow rate before merging (vehicles per second)
    c_f_after = n_after / aggr_time  # Flow rate after merging (vehicles per second)
    w_f = 0.7  # Weight for flow

    # Density calculation
    c_d_before = n_before / (n_lanes * seg0_before_merge_length)  # Density before merging (vehicles per meter)
    c_d_after = n_after / (n_lanes * seg0_after_merge_length)  # Density after merging (vehicles per meter)
    avg_density = (c_d_before + c_d_after) / 2
    
    w_d = 0.4  # Weight for density

    # Collision penalty
    r_c = -100 if collisions >= collision_occurred else 0  # Heavier penalty for more than threshold collisions

    # Dynamic Optimal Speed Calculation based on density
    if avg_density > optimal_density:  
        optimal_speed = max(60, min(90, max_speed - avg_density * 100))  
        # Penalize if density exceeds maximum acceptable level
        if avg_density > max_density:
            r_dens_penalty = -(avg_density - max_density) * 100
        else:
            r_dens_penalty = 0
    else:  
        optimal_speed = min(130, max(90, max_speed - avg_density * 50))
        r_dens_penalty = 0
    
    # Speed control reward/penalty based on deviation from dynamic optimal speed
    avg_speed_value = np.mean(avg_speed) if isinstance(avg_speed, list) else avg_speed
    
    if avg_speed_value < optimal_speed:
        r_s = -(optimal_speed - avg_speed_value) / optimal_speed  
    elif avg_speed_value > max_speed:
        r_s = -(avg_speed_value - max_speed) / max_speed  
    else:
        r_s = min((avg_speed_value - optimal_speed) / optimal_speed, 1)

    w_s = 0.15  # Weight for speed control

    # Convergence term to encourage stability between flow/density before and after merging points
    convergence_factor = abs(c_f_before - c_f_after) + abs(c_d_before - c_d_after)
    
    w_convergence = 0.2  
    r_convergence = -(convergence_factor)

    # Final reward calculation: Add stronger penalties for emissions and density imbalances
    reward = (
        epsilon_r / (epsilon_r + w_f * (c_f_before + c_f_after) + w_d * (c_d_before + c_d_after)) +
        w_s * r_s + 
        r_c + 
        w_convergence * r_convergence +
        r_dens_penalty
    )
    
    return reward

def reward_function_v7(model, n_before, n_after, avg_speed, emissions, collisions,
                       n_lanes=3, seg0_before_merge_length=575,
                       seg0_after_merge_length=500, collision_occurred=10,
                       aggr_time=60):  
    # Constants
    epsilon_r = 0.01  # Small constant to avoid division by zero
    max_speed = 130  # Maximum allowed speed in km/h
    optimal_density = 0.05  # Optimal density (vehicles per meter)
    max_density = 0.1  # Maximum acceptable density before heavy penalties
    emission_penalty_factor = 0.002  # Increased penalty factor for emissions

    # Flow rate calculation
    c_f_before = n_before / aggr_time  # Flow rate before merging (vehicles per second)
    c_f_after = n_after / aggr_time  # Flow rate after merging (vehicles per second)
    w_f = 0.5  # Weight for flow

    # Density calculation
    c_d_before = n_before / (n_lanes * seg0_before_merge_length)  # Density before merging (vehicles per meter)
    c_d_after = n_after / (n_lanes * seg0_after_merge_length)  # Density after merging (vehicles per meter)
    avg_density = (c_d_before + c_d_after) / 2
    
    w_d_before = 0.5  
    w_d_after = 0.6  # Increased weight for density after merging

    # Collision penalty
    r_c = -100 if collisions >= collision_occurred else 0  

    # Dynamic Optimal Speed Calculation based on density
    if avg_density > optimal_density:  
        optimal_speed = max(60, min(90, max_speed - avg_density * 100))  
        if avg_density > max_density:
            r_dens_penalty = -(avg_density - max_density) * 100
        else:
            r_dens_penalty = 0
    else:  
        optimal_speed = min(130, max(90, max_speed - avg_density * 50))
        r_dens_penalty = 0
    
    # Speed control reward/penalty based on deviation from dynamic optimal speed
    avg_speed_value = np.mean(avg_speed) if isinstance(avg_speed, list) else avg_speed
    
    if avg_speed_value < optimal_speed:
        r_s = -(optimal_speed - avg_speed_value) / optimal_speed  
    elif avg_speed_value > max_speed:
        r_s = -(avg_speed_value - max_speed) / max_speed  
    else:
        r_s = min((avg_speed_value - optimal_speed) / optimal_speed, 1)

    w_s = 0.25  

    # Emission penalty
    r_emissions_penalty = -emission_penalty_factor * emissions  

    # Convergence term to encourage stability between flow/density before and after merging points
    convergence_factor = abs(c_f_before - c_f_after) + abs(c_d_before - c_d_after)
    
    w_convergence = 0.15  
    r_convergence = -(convergence_factor)

    # Final reward calculation: Add stronger penalties for emissions and density imbalances
    reward = (
        epsilon_r / (epsilon_r + w_f * (c_f_before + c_f_after) + w_d_before * c_d_before + w_d_after * c_d_after) +
        w_s * r_s + 
        r_c + 
        r_emissions_penalty + 
        w_convergence * r_convergence +
        r_dens_penalty
    )
    
    return reward

def reward_function_v8(model, n_seg0, n_seg1, n_after, avg_speed, collisions, gen_rate,
                       seg0_length=275, seg1_length=300, seg_after_length=500,
                       collision_occurred=5, aggr_time=60):
    # Constants and Neutral Zones
    epsilon_r = 0.01  
    max_speed = 130  
    neutral_density_range = (0.045, 0.055)  
    neutral_flow_range = (0.9 * gen_rate / aggr_time, 1.1 * gen_rate / aggr_time)  
    
    # Flow and Density Calculations per Segment
    c_f_seg0 = n_seg0 / aggr_time  
    c_f_seg1 = n_seg1 / aggr_time  
    c_f_after = n_after / aggr_time  
    avg_flow = (c_f_seg0 + c_f_seg1 + c_f_after) / 3
    
    c_d_seg0 = n_seg0 / seg0_length  
    c_d_seg1 = n_seg1 / seg1_length  
    c_d_after = n_after / seg_after_length  
    avg_density = (c_d_seg0 + c_d_seg1 + c_d_after) / 3
    
    # Collision Penalty
    r_c = -100 if collisions >= collision_occurred else 0

    # Neutral Zone Logic for Density and Flow Penalties
    if neutral_density_range[0] <= avg_density <= neutral_density_range[1]:
        r_dens_penalty = 0
    else:
        r_dens_penalty = -(abs(avg_density - sum(neutral_density_range) / 2) * 50)

    if neutral_flow_range[0] <= avg_flow <= neutral_flow_range[1]:
        r_flow_penalty = 0
    else:
        r_flow_penalty = -(abs(avg_flow - sum(neutral_flow_range) / 2) * 50)

    # Speed Control Reward/Penalty Based on Optimal Speed
    optimal_speed = max(60, min(90, max_speed - avg_density * 100))
    
    avg_speed_value = np.mean(avg_speed) if isinstance(avg_speed, list) else avg_speed
    
    if avg_speed_value < optimal_speed:
        r_s = -(optimal_speed - avg_speed_value) / optimal_speed  
    elif avg_speed_value > max_speed:
        r_s = -(avg_speed_value - max_speed) / max_speed  
    else:
        r_s = min((avg_speed_value - optimal_speed) / optimal_speed, 1)

    # Final Reward Calculation with Normalization
    reward = (
        epsilon_r /
        (epsilon_r + avg_flow + avg_density) +
        r_s +
        r_c +
        r_dens_penalty +
        r_flow_penalty
    )
    
    return reward

def reward_function_v9(model, n_seg0, n_seg1, n_after, avg_speed, collisions, gen_rate,
                       seg0_length=275, seg1_length=300, seg_after_length=500,
                       collision_occurred=5, aggr_time=60):
    # Constants and Neutral Zones
    epsilon_r = 1e-9  
    max_speed = 130  
    neutral_density_range = (0.045, 0.055)  
    neutral_flow_range = (0.9 * gen_rate / aggr_time, 1.1 * gen_rate / aggr_time)  
    
    # Flow and Density Calculations per Segment
    c_f_seg0 = n_seg0 / aggr_time  
    c_f_seg1 = n_seg1 / aggr_time  
    c_f_after = n_after / aggr_time  
    avg_flow = (c_f_seg0 + c_f_seg1 + c_f_after) / 3
    
    c_d_seg0 = n_seg0 / seg0_length  
    c_d_seg1 = n_seg1 / seg1_length  
    c_d_after = n_after / seg_after_length  
    avg_density = (c_d_seg0 + c_d_seg1 + c_d_after) / 3
    
    # Normalize Flow and Density
    max_flow = gen_rate / aggr_time  # Maximum possible flow based on generation rate
    max_density = gen_rate / (seg0_length + seg1_length + seg_after_length)  # Approximate max density
    
    normalized_flow = (avg_flow + epsilon_r) / (max_flow + epsilon_r)
    normalized_density = (avg_density + epsilon_r) / (max_density + epsilon_r)

    # Collision Penalty
    r_c = -100 if collisions >= collision_occurred else 0

    # Neutral Zone Logic for Density and Flow Penalties
    if neutral_density_range[0] <= avg_density <= neutral_density_range[1]:
        r_dens_penalty = 0
    else:
        r_dens_penalty = -(abs(avg_density - sum(neutral_density_range) / 2) * 50)

    if neutral_flow_range[0] <= avg_flow <= neutral_flow_range[1]:
        r_flow_penalty = 0
    else:
        r_flow_penalty = -(abs(avg_flow - sum(neutral_flow_range) / 2) * 50)

    # Speed Control Reward/Penalty Based on Optimal Speed
    optimal_speed = max(60, min(90, max_speed - avg_density * 100))
    
    avg_speed_value = np.mean(avg_speed) if isinstance(avg_speed, list) else avg_speed
    
    if avg_speed_value < optimal_speed:
        r_s = -(optimal_speed - avg_speed_value + epsilon_r) / (optimal_speed + epsilon_r)
    elif avg_speed_value > max_speed:
        r_s = -(avg_speed_value - max_speed + epsilon_r) / (max_speed + epsilon_r)
    else:
        r_s = min((avg_speed_value - optimal_speed) / optimal_speed, 1)

    # Reward Stabilization with Scaling
    scaling_factor = gen_rate / 1000  # Scale reward based on vehicle generation rate
    reward_raw = (
        epsilon_r /
        (epsilon_r + normalized_flow + normalized_density) +
        r_s +
        r_c +
        r_dens_penalty +
        r_flow_penalty
    )
    
    reward_stabilized = (reward_raw + epsilon_r) / (scaling_factor + epsilon_r)

    # Final Reward Calculation with Offset for Stabilization Around Zero
    baseline_offset = 50  # Shift rewards closer to zero for interpretability
    reward_final = reward_stabilized + baseline_offset
    
    return reward_final

def reward_function_v10(model, n_seg0, n_seg1, n_after, avg_speed, collisions, gen_rate,
                        seg0_length=275, seg1_length=300, seg_after_length=500,
                        collision_occurred=5, aggr_time=60):
    # Constants and Neutral Zones
    epsilon_r = 1e-9  
    max_speed = 130  
    neutral_density_range = (0.045, 0.055)  
    neutral_flow_range = (0.9 * gen_rate / aggr_time, 1.1 * gen_rate / aggr_time)  
    n_lanes_seg0 = 3  # Number of lanes in pre-merge zone
    n_lanes_seg1 = 3  # Number of lanes in merge zone
    n_lanes_after = 2  # Number of lanes in post-merge zone
    
    # Corrected Flow Calculations per Segment
    c_f_seg0 = n_seg0 / (aggr_time * n_lanes_seg0)  # Flow rate per lane for pre-merge zone
    c_f_seg1 = n_seg1 / (aggr_time * n_lanes_seg1)  # Flow rate per lane for merge zone
    c_f_after = n_after / (aggr_time * n_lanes_after)  # Flow rate per lane for post-merge zone
    
    avg_flow = (c_f_seg0 + c_f_seg1 + c_f_after) / 3

    # Corrected Density Calculations per Segment
    c_d_seg0 = n_seg0 / (seg0_length * n_lanes_seg0)  # Density per lane for pre-merge zone
    c_d_seg1 = n_seg1 / (seg1_length * n_lanes_seg1)  # Density per lane for merge zone
    c_d_after = n_after / (seg_after_length * n_lanes_after)  # Density per lane for post-merge zone
    
    avg_density = (c_d_seg0 + c_d_seg1 + c_d_after) / 3

    # Normalize Flow and Density
    max_flow = gen_rate / aggr_time  # Maximum possible flow based on generation rate
    max_density = gen_rate / ((seg0_length + seg1_length + seg_after_length) * min(n_lanes_seg0, n_lanes_after))  
    normalized_flow = (avg_flow + epsilon_r) / (max_flow + epsilon_r)
    normalized_density = (avg_density + epsilon_r) / (max_density + epsilon_r)

    # Collision Penalty
    r_c = -100 if collisions >= collision_occurred else 0

    # Neutral Zone Logic for Density and Flow Penalties
    if neutral_density_range[0] <= avg_density <= neutral_density_range[1]:
        r_dens_penalty = 0
    else:
        r_dens_penalty = -(abs(avg_density - sum(neutral_density_range) / 2) * 50)

    if neutral_flow_range[0] <= avg_flow <= neutral_flow_range[1]:
        r_flow_penalty = 0
    else:
        r_flow_penalty = -(abs(avg_flow - sum(neutral_flow_range) / 2) * 50)

    # Speed Control Reward/Penalty Based on Optimal Speed
    optimal_speed = max(60, min(90, max_speed - avg_density * 100))
    
    avg_speed_value = np.mean(avg_speed) if isinstance(avg_speed, list) else avg_speed
    
    if avg_speed_value < optimal_speed:
        r_s = -(optimal_speed - avg_speed_value + epsilon_r) / (optimal_speed + epsilon_r)
    elif avg_speed_value > max_speed:
        r_s = -(avg_speed_value - max_speed + epsilon_r) / (max_speed + epsilon_r)
    else:
        r_s = min((avg_speed_value - optimal_speed) / optimal_speed, 1)

    # Reward Stabilization with Scaling
    scaling_factor = gen_rate / 1000  # Scale reward based on vehicle generation rate
    reward_raw = (
        epsilon_r /
        (epsilon_r + normalized_flow + normalized_density) +
        r_s +
        r_c +
        r_dens_penalty +
        r_flow_penalty
    )
    
    reward_stabilized = (reward_raw + epsilon_r) / (scaling_factor + epsilon_r)

    # Final Reward Calculation with Offset for Stabilization Around Zero
    baseline_offset = 0  # Shift rewards closer to zero for interpretability FIXME: 0 for the moment
    reward_final = reward_stabilized + baseline_offset
    
    return reward_final

def reward_function_v11(model, n_seg0, n_seg1, n_after, avg_speed, collisions, gen_rate,
                        seg0_length=275, seg1_length=300, seg_after_length=500,
                        max_collis_no_pen=5, # Maximum number of collisions before penalty
                        aggr_time=60,
                        min_reward=-835, # Based on empirical experiments 
                        max_reward=-831 # Based on empirical experiments
                        ):
    # Constants and Neutral Zones
    epsilon_r = 1e-9  
    max_speed = 130  
    neutral_density_range = (0.045, 0.055)  
    neutral_flow_range = (0.9 * gen_rate / aggr_time, 1.1 * gen_rate / aggr_time)  
    n_lanes_seg0 = 3  # Number of lanes in pre-merge zone
    n_lanes_seg1 = 3  # Number of lanes in merge zone
    n_lanes_after = 2  # Number of lanes in post-merge zone
    
    # Corrected Flow Calculations per Segment
    q_seg0 = n_seg0 / (aggr_time * n_lanes_seg0)  
    q_seg1 = n_seg1 / (aggr_time * n_lanes_seg1)  
    q_after = n_after / (aggr_time * n_lanes_after)  
    avg_flow = (q_seg0 + q_seg1 + q_after) / 3

    # Corrected Density Calculations per Segment
    k_seg0 = n_seg0 / (seg0_length * n_lanes_seg0)  
    k_seg1 = n_seg1 / (seg1_length * n_lanes_seg1)  
    k_after = n_after / (seg_after_length * n_lanes_after)  
    avg_density = (k_seg0 + k_seg1 + k_after) / 3

    # Normalize Flow and Density
    max_flow = gen_rate / aggr_time  
    max_density = gen_rate / ((seg0_length + seg1_length + seg_after_length) * min(n_lanes_seg0, n_lanes_after))  
    normalized_flow = (avg_flow + epsilon_r) / (max_flow + epsilon_r)
    normalized_density = (avg_density + epsilon_r) / (max_density + epsilon_r)

    # Collision Penalty
    r_c = -100 if collisions > max_collis_no_pen else 0

    # Neutral Zone Logic for Density and Flow Penalties
    if neutral_density_range[0] <= avg_density <= neutral_density_range[1]:
        r_dens_penalty = 0
    else:
        r_dens_penalty = -(abs(avg_density - sum(neutral_density_range) / 2) * 50)

    if neutral_flow_range[0] <= avg_flow <= neutral_flow_range[1]:
        r_flow_penalty = 0
    else:
        r_flow_penalty = -(abs(avg_flow - sum(neutral_flow_range) / 2) * 50)

    # Speed Control Reward/Penalty Based on Optimal Speed
    optimal_speed = max(60, min(90, max_speed - avg_density * 100))
    
    avg_speed_value = np.mean(avg_speed) if isinstance(avg_speed, list) else avg_speed
    
    if avg_speed_value < optimal_speed:
        r_s = -(optimal_speed - avg_speed_value + epsilon_r) / (optimal_speed + epsilon_r)
    elif avg_speed_value > max_speed:
        r_s = -(avg_speed_value - max_speed + epsilon_r) / (max_speed + epsilon_r)
    else:
        r_s = min((avg_speed_value - optimal_speed) / optimal_speed, 1)

    # Raw Reward Calculation
    scaling_factor = gen_rate / 1000  
    reward_raw = (
        epsilon_r /
        (epsilon_r + normalized_flow + normalized_density) +
        r_s +
        r_c +
        r_dens_penalty +
        r_flow_penalty
    )
    
    reward_stabilized = (reward_raw + epsilon_r) / (scaling_factor + epsilon_r)

    # Normalize Reward Between [-1, 1]
    reward_normalized = 2 * ((reward_stabilized - min_reward) / (max_reward - min_reward)) - 1

    return reward_normalized

def reward_function_v12(model, n_seg0, n_seg1, n_after, avg_speed, collisions, gen_rate, vsl,
                               seg0_length=275, seg1_length=300, seg_after_length=500,
                               collision_occurred=5, aggr_time=60,
                               min_reward=-835, max_reward=-831,
                               compliance_weight=10, extreme_speed_weight=5,
                               acceptable_deviation=10):
    # Constants and Neutral Zones
    epsilon_r = 1e-9
    max_speed = 130
    neutral_density_range = (0.045, 0.055)
    neutral_flow_range = (0.9 * gen_rate / aggr_time, 1.1 * gen_rate / aggr_time)
    n_lanes_seg0 = 3
    n_lanes_seg1 = 3
    n_lanes_after = 2

    # Corrected Flow Calculations per Segment
    q_seg0 = n_seg0 / (aggr_time * n_lanes_seg0)
    q_seg1 = n_seg1 / (aggr_time * n_lanes_seg1)
    q_after = n_after / (aggr_time * n_lanes_after)
    avg_flow = (q_seg0 + q_seg1 + q_after) / 3

    # Corrected Density Calculations per Segment
    k_seg0 = n_seg0 / (seg0_length * n_lanes_seg0)
    k_seg1 = n_seg1 / (seg1_length * n_lanes_seg1)
    k_after = n_after / (seg_after_length * n_lanes_after)
    avg_density = (k_seg0 + k_seg1 + k_after) / 3

    # Normalize Flow and Density
    max_flow = gen_rate / aggr_time
    max_density = gen_rate / ((seg0_length + seg1_length + seg_after_length) * min(n_lanes_seg0, n_lanes_after))
    normalized_flow = (avg_flow + epsilon_r) / (max_flow + epsilon_r)
    normalized_density = (avg_density + epsilon_r) / (max_density + epsilon_r)

    # Collision Penalty
    r_c = -100 if collisions >= collision_occurred else 0

    # Neutral Zone Logic for Density and Flow Penalties
    if neutral_density_range[0] <= avg_density <= neutral_density_range[1]:
        r_dens_penalty = 0
    else:
        r_dens_penalty = -(abs(avg_density - sum(neutral_density_range) / 2) * 50)

    if neutral_flow_range[0] <= avg_flow <= neutral_flow_range[1]:
        r_flow_penalty = 0
    else:
        r_flow_penalty = -(abs(avg_flow - sum(neutral_flow_range) / 2) * 50)

    # Compliance Penalty
    compliance_penalty = -compliance_weight * abs(avg_speed - vsl)

    # Extreme Speed Penalty
    extreme_speed_penalty = (
        -extreme_speed_weight * abs(avg_speed - vsl)
        if abs(avg_speed - vsl) > acceptable_deviation else 
        0
    )

    # Speed Control Reward/Penalty Based on Optimal Speed
    optimal_speed = max(60, min(90, max_speed - avg_density * 100))
    
    if avg_speed < optimal_speed:
        r_s = -(optimal_speed - avg_speed + epsilon_r) / (optimal_speed + epsilon_r)
    elif avg_speed > max_speed:
        r_s = -(avg_speed - max_speed + epsilon_r) / (max_speed + epsilon_r)
    else:
        r_s = min((avg_speed - optimal_speed) / optimal_speed, 1)

    # Raw Reward Calculation
    scaling_factor = gen_rate / 1000
    reward_raw = (
        epsilon_r /
        (epsilon_r + normalized_flow + normalized_density) +
        r_s +
        r_c +
        r_dens_penalty +
        r_flow_penalty +
        compliance_penalty +
        extreme_speed_penalty
    )

    # Normalize Reward Between [-1, 1]
    reward_stabilized = (reward_raw + epsilon_r) / (scaling_factor + epsilon_r)

    return reward_stabilized

def reward_function_v13(model, n_seg0, n_seg1, n_after, avg_speed, seg0_collisions, gen_rate, vsl,
                        seg0b_length=275, seg1b_length=300, sega_length=500,
                        collision_occurred=5, aggr_time=60,
                        compliance_weight=10, extreme_speed_weight=5,
                        acceptable_deviation=10,
                        min_reward=-834.86, max_reward=-0.44):
    # Constants
    epsilon_r = 1e-9
    max_speed = 130
    neutral_density_range = (0.045, 0.055)
    neutral_flow_range = (0.9 * gen_rate / aggr_time, 1.1 * gen_rate / aggr_time)
    n_lanes_seg0 = 3
    n_lanes_seg1 = 3
    n_lanes_after = 2

    # Normalize Inputs
    def normalize(x, x_min, x_max):
        return 2 * ((x - x_min) / (x_max - x_min)) - 1

    # Flow and Density Normalization
    q_seg0 = n_seg0 / (aggr_time * n_lanes_seg0)
    q_seg1 = n_seg1 / (aggr_time * n_lanes_seg1)
    q_after = n_after / (aggr_time * n_lanes_after)
    avg_flow = (q_seg0 + q_seg1 + q_after) / 3

    k_seg0 = n_seg0 / (seg0b_length * n_lanes_seg0)
    k_seg1 = n_seg1 / (seg1b_length * n_lanes_seg1)
    k_after = n_after / (sega_length * n_lanes_after)
    avg_density = (k_seg0 + k_seg1 + k_after) / 3

    max_flow = gen_rate / aggr_time
    max_density = gen_rate / ((seg0b_length + seg1b_length + sega_length) * min(n_lanes_seg0, n_lanes_after))

    normalized_flow = normalize(avg_flow, 0, max_flow)
    normalized_density = normalize(avg_density, 0, max_density)

    # Collision Penalty
    r_c = -100 if seg0_collisions >= collision_occurred else 0

    # Neutral Zone Logic for Density and Flow Penalties
    if neutral_density_range[0] <= avg_density <= neutral_density_range[1]:
        r_dens_penalty = 0
    else:
        r_dens_penalty = -(abs(avg_density - sum(neutral_density_range) / 2) * 50)

    if neutral_flow_range[0] <= avg_flow <= neutral_flow_range[1]:
        r_flow_penalty = 0
    else:
        r_flow_penalty = -(abs(avg_flow - sum(neutral_flow_range) / 2) * 50)

    # Compliance Penalty
    compliance_penalty = -compliance_weight * abs(avg_speed - vsl)

    # Extreme Speed Penalty
    extreme_speed_penalty = (
        -extreme_speed_weight * abs(avg_speed - vsl)
        if abs(avg_speed - vsl) > acceptable_deviation else 
        0
    )

    # Speed Control Reward/Penalty Based on Optimal Speed
    optimal_speed = max(60, min(90, max_speed - avg_density * 100))
    
    if avg_speed < optimal_speed:
        r_s = -(optimal_speed - avg_speed + epsilon_r) / (optimal_speed + epsilon_r)
    elif avg_speed > max_speed:
        r_s = -(avg_speed - max_speed + epsilon_r) / (max_speed + epsilon_r)
    else:
        r_s = min((avg_speed - optimal_speed) / optimal_speed, 1)

    # Raw Reward Calculation with Normalized Components
    reward_raw = (
        epsilon_r /
        (epsilon_r + normalized_flow + normalized_density) +
        r_s +
        r_c +
        r_dens_penalty +
        r_flow_penalty +
        compliance_penalty +
        extreme_speed_penalty
    )

    # Normalize Final Reward Between [-1, 1]
    reward_normalized = normalize(reward_raw, min_reward, max_reward)

    return reward_normalized

def reward_function_v14(model, n_seg0, n_seg1, n_after, avg_speed, seg0_collisions, gen_rate, vsl,
                        seg0b_length=275, seg1b_length=300, sega_length=500,
                        collision_occurred=5, aggr_time=60,
                        compliance_weight=10, extreme_speed_weight=5,
                        min_reward=-835, max_reward=-831,
                        disobey_weight=10, disobey_deviation=0.05):
    
    # Constants
    epsilon_r = 1e-9
    max_speed = 130
    n_lanes_seg0 = 3
    n_lanes_seg1 = 3
    n_lanes_after = 2

    # Normalize Inputs
    def normalize(x, x_min, x_max):
        return (x - x_min) / (x_max - x_min)

    # Flow and Density Calculations per Segment
    q_seg0 = n_seg0 / (aggr_time * n_lanes_seg0)
    q_seg1 = n_seg1 / (aggr_time * n_lanes_seg1)
    q_after = n_after / (aggr_time * n_lanes_after)
    avg_flow = (q_seg0 + q_seg1 + q_after) / 3

    k_seg0 = n_seg0 / (seg0b_length * n_lanes_seg0)
    k_seg1 = n_seg1 / (seg1b_length * n_lanes_seg1)
    k_after = n_after / (sega_length * n_lanes_after)
    avg_density = (k_seg0 + k_seg1 + k_after) / 3

    # Normalize Flow and Density
    max_flow = gen_rate / aggr_time
    max_density = gen_rate / ((seg0b_length + seg1b_length + sega_length) * min(n_lanes_seg0, n_lanes_after))
    normalized_flow = normalize(avg_flow, 0, max_flow)
    normalized_density = normalize(avg_density, 0, max_density)

    # Collision Penalty
    r_c = -1 if seg0_collisions >= collision_occurred else 0

    # Dynamic Neutral Zone Logic for Density and Flow Penalties
    neutral_density_range = (0.045 * avg_density / max_density, 0.055 * avg_density / max_density)
    neutral_flow_range = (0.9 * avg_flow / max_flow, 1.1 * avg_flow / max_flow)

    if neutral_density_range[0] <= avg_density <= neutral_density_range[1]:
        r_dens_penalty = 0
    else:
        r_dens_penalty = -(abs(avg_density - sum(neutral_density_range) / 2) * 50)

    if neutral_flow_range[0] <= avg_flow <= neutral_flow_range[1]:
        r_flow_penalty = 0
    else:
        r_flow_penalty = -(abs(avg_flow - sum(neutral_flow_range) / 2) * 50)

    # Disobey Factor Penalty
    deviation_threshold = disobey_deviation * vsl
    disobey_factor = (
        -disobey_weight * ((abs(avg_speed - vsl) / vsl) ** 2)
        if abs(avg_speed - vsl) > deviation_threshold else 
        0
    )

    # Speed Control Reward/Penalty Based on Optimal Speed
    optimal_speed = max(60, min(90, max_speed - avg_density * 100))

    if avg_speed < optimal_speed:
        r_s = -(optimal_speed - avg_speed + epsilon_r) / (optimal_speed + epsilon_r)
    elif avg_speed > max_speed:
        r_s = -(avg_speed - max_speed + epsilon_r) / (max_speed + epsilon_r)
    else:
        r_s = min((avg_speed - optimal_speed) / optimal_speed, 1)

    # Raw Reward Calculation with Normalized Components
    reward_raw = (
        epsilon_r /
        (epsilon_r + normalized_flow + normalized_density) +
        r_s +
        r_dens_penalty +
        r_flow_penalty +
        disobey_factor
    ) if r_c == 0 else r_c

    # Normalize Final Reward Between [-1, 1]
    # reward_normalized = normalize(reward_raw, min_reward, max_reward)

    return reward_raw

def reward_function_v15(model, n_seg0, n_after, avg_speed, 
                        o_before0_temp, o_after_temp,
                        seg0_collisions, gen_rate, vsl,
                        seg0b_length=275, seg1b_length=300, sega_length=500,
                        collision_occurred=5, aggr_time=60,
                        compliance_weight=10, emissions_weight=0.01,
                        critical_density=30, min_flow_threshold=900):
    
    # Constants
    epsilon_r = 0.1
    n_lanes_seg0 = 3
    n_lanes_seg1 = 3
    n_lanes_after = 2

    # Quad Occupancy Reward Function
    def quad_occ_reward(occupancy):
        """
        Occupancy: the percentage of time the detector was occupied by a vehicle.
        """
        if 0 < occupancy <= 12:
            return ((0.5 * occupancy) + 6) / 12
        elif 12 < occupancy < 80:
            return (-(occupancy - 80)**2) / (68**2) + 1
        else:
            return 0

    r_o = (quad_occ_reward(o_before0_temp) + quad_occ_reward(o_after_temp)) / 2
    
    w_build_up = 0.5
    w_merging_zone = 1.0
    w_post_merge = 0.7
    # Density Calculations per Segment
    k_seg0 = n_seg0 / (seg0b_length * n_lanes_seg0)
    # k_seg1 = n_seg1 / (seg1b_length * n_lanes_seg1)
    k_after = n_after / (sega_length * n_lanes_after)
    avg_density = (k_seg0 + k_after) / 2

    # Neutral Zone Logic for Density
    density_neutral_min = max(0.04, avg_density * 0.9)
    density_neutral_max = min(0.06, avg_density * 1.1)
    density_penalty_seg0 = -(abs(k_seg0 - ((density_neutral_min + density_neutral_max) / 2)) * compliance_weight) \
        if not (density_neutral_min <= k_seg0 <= density_neutral_max) else 0
    # density_penalty_seg1 = -(abs(k_seg1 - ((density_neutral_min + density_neutral_max) / 2)) * compliance_weight) \
        # if not (density_neutral_min <= k_seg1 <= density_neutral_max) else 0
    density_penalty_after = -(abs(k_after - ((density_neutral_min + density_neutral_max) / 2)) * compliance_weight) \
        if not (density_neutral_min <= k_after <= density_neutral_max) else 0
    total_density_penalty = (
        w_build_up * density_penalty_seg0 +
        # w_merging_zone * density_penalty_seg1 +
        w_post_merge * density_penalty_after
    )

    # Collision Penalty
    r_c = -seg0_collisions if seg0_collisions >= collision_occurred else 0

    # avg_speed_smoothed = pd.Series(avg_speed_seg0before_all).rolling(window=3).mean().fillna(method='bfill')
    
    # Context-Aware Compliance Penalty (Speed Limit Disobedience)
    acceptable_deviation = 0.05
    disobey_weight = 10
    if avg_speed < vsl * (1 - acceptable_deviation) or avg_speed > vsl * (1 + acceptable_deviation):
        if 0 < min_flow_threshold or avg_density > critical_density: # FIXME: Check if this is the correct condition
            # Smoothed penalty using sigmoid function
            deviation_ratio = abs(avg_speed - vsl) / vsl
            r_d = -disobey_weight / (1 + np.exp(-10 * (deviation_ratio - acceptable_deviation)))
        else:
            r_d = 0
    else:
        r_d = 0

    # # Intermediate rewards for progress toward goals (e.g., maintaining high flow or reducing density)
    # w_p = 0.01
    # r_p = w_p * max(0, (q_seg0 + q_seg1 + q_after)/3 - min_flow_threshold)

    # Raw Reward Calculation with Normalized Components
    reward_raw = (
        1.0 * r_o
        + 0.3 * total_density_penalty
        + 0.1 * r_c 
        + 0.2 * r_d
        #   + 0.41 * emissions_penalty
    )
    
    # clipped_reward = max(-1, min(1, reward_raw))
    
    return reward_raw, 0, total_density_penalty, r_c, r_d, r_o

'''--------------------------------------------------------------------------------------------------'''
""" Reward functions split to be used in CTDE context """
def calculate_disobey_penalty(avg_speed, vsl, acceptable_deviation, disobey_weight):
    # Penalize speed deviations beyond acceptable range
    if avg_speed < vsl * (1 - acceptable_deviation) or avg_speed > vsl * (1 + acceptable_deviation):
        deviation_ratio = abs(avg_speed - vsl) / vsl
        disobey_penalty = -disobey_weight / (1 + np.exp(-10 * (deviation_ratio - acceptable_deviation)))
    else:
        disobey_penalty = 0

    return disobey_penalty

def calculate_collision_penalty(seg_collisions, collision_threshold):
    # Apply penalty if collisions exceed the threshold
    collision_penalty = -seg_collisions if seg_collisions >= collision_threshold else 0
    return collision_penalty

def calculate_density_penalty(n_seg0, n_seg1, n_after, compliance_weight=10):
    total_lane_km = sum(data["length"] * data["lanes"] for data in section_data.values())
    adjusted_weights = {
        section: (data["length"] * data["lanes"]) / total_lane_km
        for section, data in section_data.items()
    }

    # Calculate densities (vehicles per km per lane)
    k_seg0 = n_seg0 / (section_data['pre_merge']['length'] * section_data['pre_merge']['lanes'])
    k_seg1 = n_seg1 / (section_data['build_up']['length'] * section_data['build_up']['lanes'])
    k_after = n_after / (section_data['after_merge']['length'] * section_data['after_merge']['lanes'])

    # Define neutral density range
    avg_density = (k_seg0 + k_seg1 + k_after) / 3
    density_neutral_min = max(0.04, avg_density * 0.9)
    density_neutral_max = min(0.06, avg_density * 1.1)

    # Calculate penalties for exceeding neutral range
    density_penalty_seg0 = -(abs(k_seg0 - ((density_neutral_min + density_neutral_max) / 2)) * compliance_weight) \
        if not (density_neutral_min <= k_seg0 <= density_neutral_max) else 0
    density_penalty_seg1 = -(abs(k_seg1 - ((density_neutral_min + density_neutral_max) / 2)) * compliance_weight) \
        if not (density_neutral_min <= k_seg1 <= density_neutral_max) else 0
    density_penalty_after = -(abs(k_after - ((density_neutral_min + density_neutral_max) / 2)) * compliance_weight) \
        if not (density_neutral_min <= k_after <= density_neutral_max) else 0

    # Weighted sum of penalties across segments
    total_density_penalty = (
        adjusted_weights['pre_merge'] * density_penalty_seg0 +
        adjusted_weights['build_up']  * density_penalty_seg1 +
        adjusted_weights['after_merge']  * density_penalty_after
    )
    
    return total_density_penalty

def calculate_flow_reward(n_seg0, n_seg1, n_after, aggr_time):
    # Calculate flow rates (vehicles per second per lane)
    q_seg0 = n_seg0 / (aggr_time * section_data['pre_merge']['lanes'])
    q_seg1 = n_seg1 / (aggr_time * section_data['build_up']['lanes'])
    q_after = n_after / (aggr_time * section_data['after_merge']['lanes'])

    # Use a custom reward function for flow based on occupancy
    def quad_occ_reward(occupancy):
        if 0 < occupancy <= 12:
            return ((0.5 * occupancy) + 6) / 12
        elif 12 < occupancy < 80:
            return (-(occupancy - 80)**2) / (68**2) + 1
        else:
            return 0

    # Convert flow rates to percentages and calculate rewards
    flow_reward_seg0 = quad_occ_reward(q_seg0 * 100)
    flow_reward_seg1 = quad_occ_reward(q_seg1 * 100)
    flow_reward_after = quad_occ_reward(q_after * 100)

    # Average flow reward across all segments
    avg_flow_reward = (flow_reward_seg0 + flow_reward_seg1 + flow_reward_after) / 3
    return avg_flow_reward

def parameterized_reward_function(n_seg0, n_seg1, n_after, 
                                  
                                  seg_collisions, avg_speed,
                                  vsl, aggr_time=60,
                                  collision_threshold=3,
                                  acceptable_deviation=0.05,
                                  disobey_weight=10):
    
    # Calculate individual components of the reward function
    flow_reward = calculate_flow_reward(n_seg0, n_seg1, n_after, aggr_time)
    
    density_penalty = calculate_density_penalty(n_seg0, n_seg1, n_after)
    
    collision_penalty = calculate_collision_penalty(seg_collisions, collision_threshold)
    
    disobey_penalty = calculate_disobey_penalty(avg_speed, vsl,
                                                acceptable_deviation,
                                                disobey_weight)

    return flow_reward, density_penalty, collision_penalty, disobey_penalty

def reward_function_v15_dynamic(weights, n_seg0, n_seg1, n_after, 
                                o_before0, o_before1, o_after,
                                avg_speed, seg0_collisions, vsl):
    # Unpack weights
    c_f, _, c_c, c_d = parameterized_reward_function(n_seg0, n_seg1, n_after, seg0_collisions, avg_speed, vsl)
    
    r_o = (quad_occ_reward(o_before0) + quad_occ_reward(o_before1) + quad_occ_reward(o_after)) / 3

    # Final Reward Calculation
    reward = (
        weights['w_flow'] * c_f +
        weights['w_density'] * r_o +
        weights['w_collision'] * c_c +
        weights['w_disobey'] * c_d
    )
    
    return reward, c_f, c_d, c_c, c_d

def sensitivity_analysis(csv_file_path):
    weights = {
        "w_flow": 0.0,
        "w_density": 0.0,
        "w_collision": 0.0,
        "w_disobey": 0.0
    }
    
    weight_range = np.linspace(0.01, 1.0, 25)  # Define weight ranges and step size
    try:
        data = pd.read_csv(csv_file_path)
        data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")
    except Exception as e:
        return f"Error reading CSV: {e}"
    
    # Initialize results list
    results = []
    
    # Iterate over all combinations of weights
    # for VSL in np.linspace(50, 140, 10):
    for vss in data["vss"].unique():
        vss_data = data[data["vss"] == vss]

        # Get needed metrics with error handling
        try:
            n_seg0 = vss_data["num_of_vehicles_before_merging_zone_(seg0)"].mean()
            n_seg1 = vss_data["num_of_vehicles_before_merging_zone_(seg1)"].mean()
            n_after = vss_data["num_of_vehicles_after_merging_zone"].mean()
            o_seg0 = vss_data["occupancy_before_merging_zone_(seg0)"].mean()
            o_seg1 = vss_data["occupancy_before_merging_zone_(seg1)"].mean()
            o_after = vss_data["occupancy_after_merging_zone"].mean()
            avg_speed = vss_data["avg_speed_in_merging_zone_(seg0)"].mean()
            seg0_collisions = vss_data["collisions_in_merging_zone_(seg0)"].mean()
            vsl = vss_data["vss"].mean()

            for w_flow in weight_range:
                for w_density in weight_range:
                    for w_collision in weight_range:
                        for w_disobey in weight_range:
                            # Update weights dynamically
                            weights["w_flow"] = w_flow
                            weights["w_density"] = w_density
                            weights["w_collision"] = w_collision
                            weights["w_disobey"] = w_disobey

                            reward, c_f, c_k, c_c, c_d = reward_function_v15_dynamic(weights, n_seg0, n_seg1, n_after, o_seg0, o_seg1, o_after, avg_speed, seg0_collisions, vsl)
                                                
                            # Append results to list
                            results.append({
                                "w_flow": w_flow,
                                "w_density": w_density,
                                "w_collision": w_collision,
                                "w_disobey": w_disobey,
                                "total_reward": reward,
                                "avg_flow": c_f,
                                "avg_density": c_k,
                                "avg_collisions": c_c,
                                "avg_disobey": c_d,
                                "avg_speed": avg_speed,
                                "vss": vss
                            })
        except KeyError as ke:
            return f"KeyError: {ke} - Check for correct CSV column names."
    return pd.DataFrame(results)

def run_heatmap(df_results):
    # Generate heatmap using pivot_table
    heatmap_data = df_results.pivot_table(
        values="total_reward",  # The column to aggregate
        index="w_flow",         # Rows of the heatmap
        columns="w_density",    # Columns of the heatmap
        aggfunc="mean"          
    )

    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".2f", cbar_kws={"shrink": 0.8})
    plt.figure(figsize=(12, 8))
    plt.title("Sensitivity Analysis Heatmap", fontsize=16)
    plt.xlabel("Weight: Density", fontsize=14)
    plt.ylabel("Weight: Flow", fontsize=14)
    
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    
    plt.tight_layout()  # Automatically adjust layout to prevent clipping
    plt.show()

def run_sensitivity_analysis_and_adjust_weights(csv_file_path):
    """
    Systematically varying weights and analyzing their impact on total rewards and individual components.
    The sensitivity analysis is done across multiple traffic scenarios (Cross-Validation)
    """
    df_results = sensitivity_analysis(csv_file_path)

    # run_heatmap(df_results)

    optimal_weights = df_results.loc[df_results.groupby("vss")["total_reward"].idxmax()]
    average_weights = optimal_weights[["w_flow", "w_density", "w_collision", "w_disobey"]].mean()
    logging.debug(f"Optimal weigths:\n{optimal_weights}")
    logging.debug(f"Average weigths:\n{average_weights}")
    sns.pairplot(df_results, vars=['w_flow', 'w_density', 'w_collision', 'w_disobey', 'total_reward'], diag_kind='kde')
    # plt.suptitle('Sensitivity Analysis Result', y=1.02)
    # plt.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Unit Testing with Predefined Values """
def correlation_heatmap(folder_to_store, csv_file_pattern):
    """
    Generate a heatmap (by using Seaborn) of the correlation matrix for the reward function evaluation metrics.
    This function loads multiple CSV files, concatenates them, and generates a heatmap based on the combined data.
    
    Ref.: Waskom, M. L. (2021). Seaborn: statistical data visualization. Journal of Open Source Software, 6(60), 3021.
    """
    
    # Define the directory where CSV files are stored
    csv_dir = os.path.join(os.getcwd(), f'eval\\rew_func\\{folder_to_store}')
    
    # Use glob to find all files that match the pattern
    csv_paths = glob.glob(os.path.join(csv_dir, f"{csv_file_pattern}*.csv"))
    
    if not csv_paths:
        logging.error("No files found matching the pattern.")
        return
    
    # Load all matching CSV files into a list of DataFrames
    dfs = [pd.read_csv(csv_path) for csv_path in csv_paths]
    
    # Concatenate all data into one DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Calculate the correlation matrix
    correlation_matrix = combined_df.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 6))

    # Generate a heatmap using seaborn
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, vmin=-1, vmax=1)

    # Add title and labels
    plt.title('Correlation of Reward with Other Metrics', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Show the plot
    plt.tight_layout()
    # plt.show()
    plt.savefig(csv_dir + f"/correlation_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

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

def plot_rewards(rewards_by_veh_gen, output_dir, window_size=50):
    """
    Plots rewards over time for each vehicle generation rate with a smoothed overlay.

    Args:
        rewards_by_veh_gen (dict): A dictionary where keys are veh_gen_per_hour and values are lists of rewards.
        output_dir (str): Directory to save the reward plots.
        window_size (int): Window size for the moving average.
    """
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through each vehicle generation rate and its corresponding rewards
    for veh_gen_rate, rewards in rewards_by_veh_gen.items():
        # Calculate moving average for smoothing
        rewards = np.array(rewards)
        moving_avg = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')

        # Plot raw rewards
        plt.figure(figsize=(10, 6))
        plt.plot(rewards, label=f"Veh Gen Rate: {veh_gen_rate}", color="blue", alpha=0.6)
        
        # Overlay moving average
        plt.plot(range(window_size - 1, len(rewards)), moving_avg, label="Smoothed (Moving Avg)", color="red", linewidth=2)

        # Add labels, title, legend, and grid
        plt.xlabel("Time Steps")
        plt.ylabel("Reward")
        plt.title(f"Reward Over Time for Veh Gen Rate {veh_gen_rate}")
        plt.legend()
        plt.grid(True)

        # Save the plot as a PNG file
        plot_path = os.path.join(output_dir, f"reward_plot_{veh_gen_rate}.png")
        plt.savefig(plot_path)
        plt.close()

    logging.info(f"Reward plots saved to {output_dir}")

def calculate_moving_average(data, column, window_size=50):
    return data[column].rolling(window=window_size).mean()

def reward_function_calibration(veh_gen_per_hour, sumo_port, folder_to_store):
    def is_within_radar_position(position, lane_id):
        radar_position = -100  # The x-coordinate for radar detection
        if lane_id in seg_0_before:
            vehicle_x_position = position[0]
            tolerance = 20.0
            if abs(vehicle_x_position - radar_position) <= tolerance:
                return True
        return False

    def get_mean_speeds_from_indloops():
        before_A, before_B, before_C, before_D, after_A = [], [], [], [], []
        speeds_idx = 0
        temp = 0
        for ind_loop_id in loops_beforeA:
            speed = traci.inductionloop.getLastStepMeanSpeed(ind_loop_id)
            if speed > 0:
                temp += speed * 3.6
                speeds_idx += 1
            before_A.append(temp / speeds_idx) if speeds_idx > 0 else None
        
        speeds_idx = 0
        temp = 0
        for ind_loop_id in loops_beforeB:
            speed = traci.inductionloop.getLastStepMeanSpeed(ind_loop_id)
            if speed > 0:
                temp += speed * 3.6
                speeds_idx += 1
            before_B.append(temp / speeds_idx) if speeds_idx > 0 else None

        speeds_idx = 0
        temp = 0
        for ind_loop_id in loops_beforeC:
            speed = traci.inductionloop.getLastStepMeanSpeed(ind_loop_id)
            if speed > 0:
                temp += speed * 3.6
                speeds_idx += 1
            before_C.append(temp / speeds_idx) if speeds_idx > 0 else None
        
        speeds_idx = 0
        temp = 0
        for ind_loop_id in loops_beforeD:
            speed = traci.inductionloop.getLastStepMeanSpeed(ind_loop_id)
            if speed > 0:
                temp += speed * 3.6
                speeds_idx += 1
            before_D.append(temp / speeds_idx) if speeds_idx > 0 else None
        
        speeds_idx = 0
        temp = 0
        for ind_loop_id in loops_after:
            speed = traci.inductionloop.getLastStepMeanSpeed(ind_loop_id)
            if speed > 0:
                temp += speed * 3.6
                speeds_idx += 1
            after_A.append(temp / speeds_idx) if speeds_idx > 0 else None

        return after_A, before_A, before_B, before_C, before_D

    model = "DQN"
    new_speed_limit = 50
    old_speed_limit = new_speed_limit
    speed_limit_run = 3 
    speed_limits = 10
    mean_speeds_by_veh_before0 = []
    model_name = 'REW_TST'
    create_sumocfg(model_name, 1)
    rewards = []
    global avg_speed_seg0before_all
    # mock_daily_pattern_test = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]
    mock_daily_pattern_test = [veh_gen_per_hour] * speed_limit_run * speed_limits
    # multiply_factor = 15
    # mock_daily_pattern_test = [x * multiply_factor for x in mock_daily_pattern_test]

    flow_generation_wrapper(model_name, 0, 1, mock_daily_pattern_test)

    try:
        sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', sumoExecutable)
        sumo_process = subprocess.Popen([sumoBinary, "-c", f"./traffic_environment/sumo/3_2_merge_{model_name}_0.sumocfg", '--start'] 
                                         + ["--default.emergencydecel=7"]
                                        #  + ["--emergencydecel.warning-threshold=1"]
                                         + ['--random-depart-offset=3600']
                                         + ["--remote-port", str(sumo_port)] 
                                         + ["--quit-on-end"] 
                                         + ["--no-warnings"]
                                           ,
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)
        traci.init(sumo_port)

        # Init the allowed Max Speed 
        # [[traci.lane.setMaxSpeed(segId, new_speed_limit / 3.6) for segId in edgeId] for edgeId in [seg_1_before]]
        # [[traci.lane.setMaxSpeed(segId, new_speed_limit / 3.6) for segId in edgeId] for edgeId in [seg_0_before]]
        [[traci.lane.setMaxSpeed(segId, new_speed_limit / 3.6) for segId in edgeId] for edgeId in [seg_0_before, seg_1_before]]
        
        # n_before_all = 0
        # n_before_all_avg = []
        seconds_passed = 0
        seconds_passed_ = 0
        aggregation_interval = 60 # seconds
        csv_dir = os.path.join(os.getcwd(), f'eval\\rew_func\\{folder_to_store}')
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        csv_path = os.path.join(csv_dir, f"RewFuncCalib_VehpHr_{veh_gen_per_hour}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        emiss_before0_avg = 0
        vehicles_on_road = []
        n_before0_avg = 0
        n_before1_avg = 0
        n_after_avg = 0
        n_before0 = []
        n_before1 = []
        n_after = []
        o_before0_avg = 0
        o_before1_avg = 0
        o_after_avg = 0
        o_before0 = []
        o_before1 = []
        o_after = []
        o_after_temp = 0
        o_before0_temp = 0
        n_before0_temp = 0
        # n_before1_temp = 0
        n_after_temp = 0
        mean_speeds_by_veh_after_temp = 0
        mean_speeds_by_veh_before0_temp = []
        avg_speed_seg0before_all = []
        mean_speeds_by_veh_after = []
        mean_speeds_by_veh_before0 = []
        mean_speeds_by_veh_before_A = []
        mean_speeds_by_veh_before_B = []
        mean_speeds_by_veh_before_C = []
        mean_speeds_by_veh_before_D = []
        
        avg_speed_seg0before = 0
        collisions_before0 = 0
        vehicles_on_road = False
        
        # Subscribe to edge data
        # all_edges_this_simu = ["seg_1_before","seg_0_before","seg_0_after"]
        for edge in edges:
            traci.edge.subscribe(edge, [
                tc.LAST_STEP_VEHICLE_NUMBER, 
                # tc.LAST_STEP_OCCUPANCY, 
                tc.LAST_STEP_MEAN_SPEED, 
                tc.VAR_CO2EMISSION])

        """ Subscribe to occupancy data for induction loops
        Purpose: Typically used to detect the presence of vehicles at specific points on the road.
        Data: Provides point-based occupancy data, which indicates whether a vehicle is present at a specific location.
        Use Case: Suitable for scenarios where you need to know vehicle presence at particular points, such as at intersections or specific road segments.
        """
        # for loop_group in loops_before:
        #     for loop_id in loop_group:
        #         traci.inductionloop.subscribe(loop_id, [tc.LAST_STEP_OCCUPANCY])
        # for loop_id in loops_after:
        #     traci.inductionloop.subscribe(loop_id, [tc.LAST_STEP_OCCUPANCY])

        """ Subscribe to data for lane area detectors
        Purpose: monitor a section of the road, providing aggregated data over an area.
        Data: Offers occupancy data over a length of the road, giving a broader picture of traffic density.
        Use Case: Ideal for monitoring traffic flow over longer stretches, such as highway segments or merging areas.
        """
        for detector_id in detectors_before + detectors_after:
            traci.lanearea.subscribe(detector_id, [
                tc.LAST_STEP_OCCUPANCY, 
                tc.LAST_STEP_VEHICLE_NUMBER, 
                tc.LAST_STEP_MEAN_SPEED,
                tc.LAST_STEP_VEHICLE_HALTING_NUMBER
                ])
        
        """
        Purpose: point-based detection
        """
        for ind_loop_id in loops_beforeA + loops_beforeB + loops_beforeC + loops_beforeD:
            traci.inductionloop.subscribe(ind_loop_id, [tc.LAST_STEP_OCCUPANCY,
                                                        # tc.LAST_STEP_VEHICLE_NUMBER, 
                                                        # tc.LAST_STEP_MEAN_SPEED
                                                        ])
        

        # ego_vehicle = "ego_vehicle"
        # traci.vehicle.subscribeContext(ego_vehicle, tc.CMD_GET_VEHICLE_VARIABLE, 100.0, [tc.VAR_SPEED])

        with open(csv_path, mode='w', newline='') as csvfile:
            fieldnames = ['Minute', 'Reward', 
                          'Num of vehicles before merging zone (seg0)',
                        #   'Num of vehicles before merging zone (seg1)',
                          'Num of vehicles after merging zone',
                          'Occupancy before merging zone (seg0)',
                        #   'Occupancy before merging zone (seg1)',
                          'Occupancy after merging zone',
                          'Emissions in merging zone (seg0)',
                          'Avg speed in merging zone (seg0)',
                          'VSS',
                          'Vehicles generated each hour',
                          'Collisions in merging zone (seg0)',
                          'Reward: occupancy',
                          'Reward: flow',
                          'Penalty: density',
                          'Penalty: compliance',
                          'Penalty: disobey']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header
            writer.writeheader()

            # while seconds_passed < 60 * aggregation_interval * len(mock_daily_pattern_test):
            while traci.simulation.getMinExpectedNumber() > 0:
                try:
                    traci.simulationStep()
                except (FatalTraCIError, TraCIException):
                    logging.debug("SUMO failed from reward_function_calibration")
                
                edge_data = {edge: traci.edge.getSubscriptionResults(edge) for edge in edges}

                if vehicles_on_road == False:
                    n_before_all_temp = 0
                    for i in range(len(edges)-2):
                        n_before_all_temp += edge_data[f'seg_{i}_before'][tc.LAST_STEP_VEHICLE_NUMBER]
                    n_before_all_temp += edge_data['seg_0_after'][tc.LAST_STEP_VEHICLE_NUMBER]
                    if n_before_all_temp > 0:
                        vehicles_on_road = True
                
                if vehicles_on_road:
                    # Retrieve occupancy data from lane area detectors
                    mean_speeds_by_veh_before0_temp_ = 0
                    speeds_idx = 0
                    for detector_id in detectors_before:
                        o_before0_temp += traci.lanearea.getSubscriptionResults(detector_id)[tc.LAST_STEP_OCCUPANCY]
                        n_before0_temp += traci.lanearea.getSubscriptionResults(detector_id)[tc.LAST_STEP_VEHICLE_NUMBER]
                    n_before0_temp = n_before0_temp / len(detectors_before)
                    
                    mean_speeds_after_A, mean_speeds_before_A, mean_speeds_before_B, mean_speeds_before_C, mean_speeds_before_D = ()
                    get_mean_speeds_from_indloops
                    speeds_idx = 0
                    for detector_id in detectors_after:
                        o_after_temp += traci.lanearea.getSubscriptionResults(detector_id)[tc.LAST_STEP_OCCUPANCY]
                        n_after_temp += traci.lanearea.getSubscriptionResults(detector_id)[tc.LAST_STEP_VEHICLE_NUMBER]
                        mean_speeds_by_veh_after_temp_ = traci.lanearea.getSubscriptionResults(detector_id)[tc.LAST_STEP_MEAN_SPEED] * 3.6
                        if traci.lanearea.getSubscriptionResults(detector_id)[tc.LAST_STEP_VEHICLE_NUMBER] < 1 and mean_speeds_by_veh_after_temp_ > 0:
                            mean_speeds_by_veh_after_temp += mean_speeds_by_veh_after_temp_
                            speeds_idx += 1
                    o_after_temp = o_after_temp / len(detectors_after)
                    n_after_temp = n_after_temp / len(detectors_after)
                    mean_speeds_by_veh_after_temp = mean_speeds_by_veh_after_temp / speeds_idx if speeds_idx > 0 else mean_speeds_by_veh_after_temp

                    # n_before0_temp = edge_data['seg_0_before'][tc.LAST_STEP_VEHICLE_NUMBER]
                    # n_before1_temp = edge_data['seg_1_before'][tc.LAST_STEP_VEHICLE_NUMBER]
                    # n_after_temp = edge_data['seg_0_after'][tc.LAST_STEP_VEHICLE_NUMBER]
                    n_before0.append(n_before0_temp) if n_before0_temp > 0 else None
                    # n_before1.append(n_before1_temp) if n_before1_temp > 0 else None
                    n_after.append(n_after_temp) if n_after_temp > 0 else None

                    o_before0.append(o_before0_temp) if o_before0_temp > 0 else None
                    # o_before1.append(o_before1_temp) if o_before1_temp > 0 else None
                    o_after.append(o_after_temp) if o_after_temp > 0 else None

                    # mean_speeds_by_veh_before0.append(mean_speeds_by_veh_before0_temp) if mean_speeds_by_veh_before0_temp > 0 else None
                    # mean_speeds_by_veh_after.append(mean_speeds_by_veh_after_temp) if mean_speeds_by_veh_after_temp > 0 else None

                    for vehId in traci.simulation.getCollidingVehiclesIDList():
                        if traci.vehicle.getRoadID(vehId) == "seg_0_before":
                            collisions_before0 += 1
                       
                    # Adjust vehicle generation rate based on traffic volume
                    if seconds_passed % aggregation_interval == 0 and seconds_passed > 0: # each minute                       
                        if (len(n_after) > 0 or len(n_before0) > 0) and vehicles_on_road:
                            """ The use of average speed getLastStepMeanSpeed() is susceptible to outliers caused 
                            by non-compliant vehicles (e.g., vehicles traveling significantly slower or faster than the VSL). 
                            These outliers can distort the average and lead to incorrect inferences by the DRL agent. """
                            # mean_speeds_by_veh_after.append(edge_data['seg_0_after'][tc.LAST_STEP_MEAN_SPEED] * 3.6)
                            # mean_speeds_by_veh_before0.append(edge_data['seg_0_before'][tc.LAST_STEP_MEAN_SPEED] * 3.6)
                            # mean_speeds_by_veh_before1.append(edge_data['seg_1_before'][tc.LAST_STEP_MEAN_SPEED] * 3.6)

                            """ Robust metrics, such as the median speed or the interquartile range (IQR), are less sensitive to 
                            extreme values and provide a more accurate representation of typical traffic behavior."""
                            if len(mean_speeds_before_C) > 0:
                                avg_speed_seg0before = np.median(mean_speeds_before_C)
                                avg_speed_seg0before_all.append(avg_speed_seg0before) if avg_speed_seg0before > 0 else None
                                mean_speeds_before_C.clear()
                            # q75, q25 = np.percentile(mean_speeds_by_veh_before0, [75, 25])
                            # avg_speed_in_seg0before = q75 - q25

                            # emissions_before0.append(traci.edge.getCO2Emission("seg_0_before"))
                            emissions_temp = edge_data['seg_0_before'][tc.VAR_CO2EMISSION]
                            if emissions_temp > 0:
                                emiss_before0_avg = np.average(emissions_temp)
                            if len(n_before0) > 0:
                                n_before0_avg = np.average(n_before0)
                            # if len(n_before1) > 0:
                            #     n_before1_avg = np.average(n_before1)
                            if len(n_after) > 0:
                                n_after_avg = np.average(n_after)
                            if len(o_before0) > 0:
                                o_before0_avg = np.average(o_before0)
                            if len(o_after) > 0:
                                o_after_avg = np.average(o_after)

                            reward, r_f, _, r_c, r_d, r_o = reward_function_v15(None, n_before0_avg, n_after_avg, 
                                                                                o_before0_avg, o_after_avg, 
                                                                                avg_speed_seg0before, collisions_before0, veh_gen_per_hour, new_speed_limit)
                            rewards.append(reward)

                            writer.writerow({
                                'Minute': seconds_passed / aggregation_interval,
                                'Reward': reward,
                                'Num of vehicles before merging zone (seg0)': n_before0_avg,
                                # 'Num of vehicles before merging zone (seg1)': n_before1_avg,
                                'Num of vehicles after merging zone': n_after_avg,
                                'Occupancy before merging zone (seg0)': o_before0_avg,
                                # 'Occupancy before merging zone (seg1)': o_before1_avg,
                                'Occupancy after merging zone': o_after_avg,
                                'Emissions in merging zone (seg0)': emiss_before0_avg,
                                'Avg speed in merging zone (seg0)': avg_speed_seg0before,
                                'VSS': new_speed_limit,
                                'Vehicles generated each hour': veh_gen_per_hour,
                                'Collisions in merging zone (seg0)': collisions_before0,
                                "Reward: occupancy": r_o,
                                "Reward: flow": r_f,
                                # "Penalty: density": r_k,
                                "Penalty: compliance": r_c,
                                "Penalty: disobey": r_d
                            })
                            # TODO: update also fieldnames in case of adding/removing a field from the csv

                            if (seconds_passed % (3600 * speed_limit_run) <= (seconds_passed_ / 3)) and seconds_passed_ > (3600 * speed_limit_run):
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

                        elif (len(n_after) == 0 or len(n_before0) == 0) and not vehicles_on_road:
                            logging.warning("No vehicles detected in the simulation.")
                        elif (len(n_after) == 0 or len(n_before0) == 0) and vehicles_on_road:
                            seconds_passed -= 1
                            seconds_passed_ -= 1

                        # mean_speeds_by_veh_before0.clear()
                        mean_speeds_by_veh_after.clear()

                    n_before0_temp = 0
                    n_after_temp = 0
                    o_before0_temp = 0
                    o_after_temp = 0
                    emissions_temp = 0
                    mean_speeds_by_veh_before0_temp = []
                    mean_speeds_by_veh_after_temp = 0
                        
                    seconds_passed += 1
                    seconds_passed_ += 1
        try:
            traci.close()
        except Exception as e:
            logging.warning(f"Error while closing TraCI: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during calibration: {e}")

    return rewards, mean_speeds_by_veh_before0

def merge_dataset(sub_folder):
    # Set the directory path where your CSV files are located
    dir_path = os.path.join(os.getcwd(), f'eval\\rew_func\\{sub_folder}')

    # Collect all CSV files matching the pattern
    csv_files = glob.glob(os.path.join(dir_path, 'RewFuncCalib_VehpHr_*.csv'))

    # Ensure there are CSV files to process
    if not csv_files:
        logging.error("No CSV files found.")
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
            filtered_files = [f for f in csv_files if extract_info(f)[1].strftime('%Y%m%d') == date_str]

            if not filtered_files:
                logging.error("No relevant CSV files found.")
            else:
                # Merge all filtered CSV files into one DataFrame
                merged_df = pd.concat([pd.read_csv(f) for f in filtered_files])

                # Create the output filename with merged pattern
                output_filename = f"RewFuncCalib_VehpHr_MERGED_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                output_path = os.path.join(dir_path, output_filename)

                # Save the merged DataFrame to a new CSV file
                merged_df.to_csv(output_path, index=False)
                
                logging.info(f"Merged CSV saved as: {output_path}")

        except ValueError as e:
            logging.error(e)

def wait_for_files(csv_folder, expected_files):
    while True:
        curr_dir = os.path.join(os.getcwd(), f'eval\\rew_func\\{csv_folder}')
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)
        existing_files = os.listdir(curr_dir)
        if 'reward_func_plot' in existing_files:
            existing_files.remove('reward_func_plot')
        stripped_existing_files = ['_'.join(file.split('_')[:3]) + '.csv' for file in existing_files]
        if all(file in stripped_existing_files for file in expected_files):
            break
        time.sleep(1)  # Wait for 1 second before checking again

def parallel_simulation(veh_gen_per_hour, sumo_port, csv_folder):
    rewards, avg_speeds = reward_function_calibration(veh_gen_per_hour, sumo_port, csv_folder)
    return veh_gen_per_hour, rewards, avg_speeds

def on_task_complete(result):
    # Callback function executed when a process completes
    logging.info(f"Task completed")

def reward_function_simulation(csv_folder):
    rewards_by_veh_gen = {}
    mp.freeze_support()

    i = 0
    while i <= 3:
        if i == 0:
            veh_gen_rates_ports = [(2000, 9000),(2250, 9001),(2500, 9002),(2750, 9003)]
        elif i == 1:
            veh_gen_rates_ports = [(3000, 9000),(3250, 9001),(3500, 9002)]
        elif i == 2:
            veh_gen_rates_ports = [(3750, 9000),(4000, 9001)]
        elif i == 3:
            veh_gen_rates_ports = [(4250, 9000)]

        with mp.Pool(processes=len(veh_gen_rates_ports)) as pool:
            results = []
            for rate, port in veh_gen_rates_ports:
                result = pool.apply_async(
                    parallel_simulation,
                    args=(rate, port, csv_folder),
                    callback=on_task_complete
                )
                results.append(result)

            # Collect results from all tasks
            rewards_by_veh_gen = {}
            for result in results:
                veh_gen_rate, rewards, avg_speeds = result.get()  # Get all return values from each process
                rewards_by_veh_gen[veh_gen_rate] = rewards
        i += 1

    merge_dataset(csv_folder)
    # correlation_heatmap(csv_folder, "RewFuncCalib_VehpHr_MERGED")
    # plot_rewards(rewards_by_veh_gen, os.path.join(os.getcwd(), f'eval\\rew_func\\{csv_folder}\\reward_plots'))

# NOTE: this function is not working as expected. FIXME: to be investigated further
def run_analysis_and_adjust_weights(csv_file_path):
    def adjust_weights_based_on_vif(X, vif_threshold=10):
        """
        Adjust weights dynamically based on VIF values.
        Features with high VIF are penalized but not completely excluded.
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        # Initialize weights dictionary
        weights = {}
        for feature, vif in zip(vif_data["Feature"], vif_data["VIF"]):
            if vif < vif_threshold:
                # Assign higher weights for lower VIFs
                weights[feature] = max(1 / vif, 0.05)
            else:
                # Assign a small but non-zero weight for high VIFs
                weights[feature] = max(0.1 / vif, 0.01)

        return weights

    """
    Perform sensitivity analysis, check Durbin-Watson statistic,
    and adjust weights dynamically based on VIF with fixes.
    """
    # Load data
    df = pd.read_csv(csv_file_path)

    # Add lagged variables or first differences
    df['lagged_flow'] = df['Reward: flow'].shift(1)
    df['lagged_density'] = df['Penalty: density'].shift(1)
    df['lagged_flow_2'] = df['Reward: flow'].shift(2)
    df['lagged_density_2'] = df['Penalty: density'].shift(2)

    # Drop rows with NaN values introduced by lagging
    df.dropna(inplace=True)

    # Select relevant features
    X = df[['Reward: flow', 'Penalty: density', 'lagged_flow', 'lagged_density', 'lagged_flow_2', 'lagged_density_2']]
    y = df['Reward']

    # Fit OLS model
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    # Check Durbin-Watson statistic: check residuals for patterns or autocorrelation using diagnostics like Durbin-Watson
    dw_stat = durbin_watson(model.resid)
    logging.debug(f"Durbin-Watson Statistic: {dw_stat}")

    if dw_stat < 1.5 or dw_stat > 2.5:
        logging.debug("Autocorrelation detected! Consider adding more lagged variables or transformations.")

    # Adjust weights dynamically based on VIF
    def adjust_weights_based_on_vif(X, vif_threshold=10):
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

        weights = {}
        for feature, vif in zip(vif_data["Feature"], vif_data["VIF"]):
            if vif < vif_threshold:
                weights[feature] = max(1 / vif, 0.05)
            else:
                weights[feature] = max(0.1 / vif, 0.01)

        return weights

    weights = adjust_weights_based_on_vif(X)
    
    logging.debug(f"Adjusted Weights Based on VIF:\n{weights}")

def linear_regression_multicollinearity(csv_file):
    df = pd.read_csv(csv_file)
    
    if "Penalty: compliance" in df.columns:
        df = df.drop(columns=["Penalty: compliance"])

    # Select relevant features
    X = df[['Reward: flow', 'Penalty: density', 'Penalty: disobey']]
    y = df['Reward']

    # Check correlation matrix for highly correlated variables
    corr_matrix = X.corr()
    logging.debug("Correlation Matrix:\n", corr_matrix)

    # Drop one of the highly correlated variables if necessary (e.g., Penalty: disobey)
    if abs(corr_matrix.loc['Penalty: density', 'Penalty: disobey']) > 0.8:
        X = X.drop(columns=['Penalty: disobey'])

    # Add constant for intercept
    X = sm.add_constant(X)

    # Fit OLS regression model
    model = sm.OLS(y, X).fit()
    logging.debug(model.summary())

if __name__ == '__main__':
    csv_folder = datetime.now().strftime('%Y%m%d_%H%M%S')
    # reward_function_simulation(csv_folder)

    static_csv = os.path.join(os.getcwd(), f"eval\\rew_func\\20241126_233513\\RewFuncCalib_VehpHr_MERGED_20241127_000334.csv")
    # merge_dataset('20241130_221015')
    # dynamic_csv = os.path.join(os.getcwd(), f'eval\\rew_func\\{csv_folder}\\RewFuncCalib_VehpHr_MERGED_*.csv') # FIXME: add wildcard choosing 
    # run_sensitivity_analysis_and_adjust_weights(static_csv)

    # linear_regression_multicollinearity(static_csv)
    # run_analysis_and_adjust_weights(static_csv)

    rewards, avg_speeds = reward_function_calibration(3000, 9011, csv_folder) # NOTE: for testing purposes
