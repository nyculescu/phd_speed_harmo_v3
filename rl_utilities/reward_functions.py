import math

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

def reward_co2_avgspeed(emissions_t, emissions_t_minus_1, speed_t, speed_t_minus_1, k1=-0.2, b1=0.9, k3=-0.1, b3=0.9):
    # Normalize emissions
    normalized_emissions_t = emissions_t / 30000000  # Example normalization factor
    normalized_emissions_t_minus_1 = emissions_t_minus_1 / 30000000

    # Calculate emission reduction component
    emission_reduction = k1 * (normalized_emissions_t - b1 * normalized_emissions_t_minus_1)
    
    # Calculate traffic flow efficiency component
    flow_efficiency = k3 * (speed_t - b3 * speed_t_minus_1)
    
    # Total reward
    reward = emission_reduction + flow_efficiency
    
    # Clip reward to prevent extreme values
    reward = max(min(reward, 100), -100)
    
    return reward