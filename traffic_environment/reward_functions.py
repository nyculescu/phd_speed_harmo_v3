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
from road import seg_1_before

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
    w_s = 0.05 # Weight for speed
    
    # Reward calculation
    reward = Q_out * penalty_accumulation - w_e * emissions - w_s * max(target_speed - avg_speed, 0) ** 2
    
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

def reward_function_calibration():
    func_rew_wrapper = reward_function_v2
    speed_limit = 90 # FIXME: the vehicles don't obey to this speed limit. To be investigated
    model_name = 'REW_TST'
    create_sumocfg(model_name, 1)
    mock_daily_pattern_test = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]
    multiply_factor = 15
    mock_daily_pattern_test = [x * multiply_factor for x in mock_daily_pattern_test]
    flow_generation_wrapper(model_name, 0, 1, mock_daily_pattern_test)
    port = 9000
    sumoBinary = os.path.join(os.environ['SUMO_HOME'], 'bin', sumoExecutable)
    sumo_process = subprocess.Popen([sumoBinary, "-c", f"./traffic_environment/sumo/3_2_merge_{model_name}_0.sumocfg", '--start'] 
                                         + ["--default.emergencydecel=7"]
                                        #  + ["--emergencydecel.warning-threshold=1"]
                                         + ['--random-depart-offset=3600']
                                         + ["--remote-port", str(port)] 
                                         + ["--quit-on-end"] 
                                           ,
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE)
    traci.init(port)
    
    n_before_all = 0
    seconds_passed = 0
    aggregation_interval = 60 # seconds
    csv_path = os.path.join(os.getcwd(), 'logs/') + 'rew_func_sim.csv'
    emissions = []
    avg_speed = []
    vehicles_on_road = []
    n_before_avg = []
    n_after_avg = []
    n_before_all_avg = []
    
    with open(csv_path, mode='w', newline='') as csvfile:
        fieldnames = ['Minute', 'Reward', 'Vehicles before', 'Vehicles after', 'Emissions', 'Avg Speed', 'Vehicles on road']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header
        writer.writeheader()
        
        for segment in [seg_1_before]:
            [traci.lane.setMaxSpeed(segId, speed_limit / 3.6) for segId in segment]

        while seconds_passed < (60 * aggregation_interval * len(mock_daily_pattern_test)) + (60 * aggregation_interval * 3):
            traci.simulationStep()
            
            for i in range(10):
                n_before_all += traci.edge.getLastStepVehicleNumber(f"seg_{i+1}_before")
            n_before = traci.edge.getLastStepVehicleNumber("seg_0_before")
            n_after = traci.edge.getLastStepVehicleNumber("seg_0_after")
            n_before_avg.append(n_before)
            n_after_avg.append(n_after)
            n_before_all_avg.append(n_before_all)
            collisions = traci.simulation.getCollidingVehiclesIDList()

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

                if (n_after >= 1 or n_before_temp >= 1) and n_before_all_temp >= 1:
                    reward = func_rew_wrapper(n_before_temp, n_after_temp, emissions_temp, avg_speed_temp, len(collisions))
                    logging.debug(f"Minute {seconds_passed / aggregation_interval}; "
                        f"Reward: {reward}; "
                        f"Vehicles before: {n_before_temp:.2f}; "
                        f"Vehicles after: {n_after_temp:.2f}; "
                        f"Emissions: {emissions_temp:.2f}; "
                        f"Avg Speed: {avg_speed_temp:.2f}; "
                        f"Vehicles on road: {vehicles_on_road_temp:.2f}"
                        )
                    writer.writerow({
                        'Minute': seconds_passed / aggregation_interval,
                        'Reward': reward,
                        'Vehicles before': n_before_temp,
                        'Vehicles after': n_after_temp,
                        'Emissions': emissions_temp,
                        'Avg Speed': avg_speed_temp,
                        'Vehicles on road': vehicles_on_road_temp
                    })
                    avg_speed.clear()
                    emissions.clear()
                    vehicles_on_road.clear()
                    n_before_avg.clear()
                    n_after_avg.clear()
                    n_before_all_avg.clear()

                    for segment in [seg_1_before]:
                        [traci.lane.setMaxSpeed(segId, speed_limit / 3.6) for segId in segment]
                elif n_after_temp == 0 and n_before_all_temp == 0:
                    logging.warning("No vehicles detected in the simulation.")
                elif n_after_temp == 0 and n_before == 0 and n_before_all_temp > 0:
                    pass
                else:
                    pass
                
            seconds_passed += 1

    traci.close()    

# Run the unit tests using TestLoader instead of makeSuite
if __name__ == '__main__':
    # test_and_tune_reward_function()
    # unittest.main()

    reward_function_calibration()
