max_emissions = 3000000 # empirical data after running the simulation at max capacity

def normalize(value, min_value, max_value):
    """Normalize a value between min_value and max_value to a range [0, 1]."""
    norm_val = (value - min_value) / (max_value - min_value) if (min_value <= value <= max_value) else 1
    return norm_val

''' This function rewards occupancy rates around 12% the most, with decreasing rewards for both lower and higher occupancies.
    - low occupancy rates (0-12%) => increases from 0.5 at x=0 to 1 at x=12
    - medium to high occupancy rates (12-80%) => decreases from 1 at x=12 to 0 at x=80
    - very low (≤0) or very high (≥80) occupancy rates => reward is 0
    The function is continuous at x=12, where both pieces evaluate to 1. However, it is not differentiable at x=12 due to the change in slope.
    Ref.: Kidando, E., Moses, R., Ozguven, E. E., & Sando, T. (2017). Evaluating traffic congestion using the traffic occupancy and speed distribution relationship: An application of Bayesian Dirichlet process mixtures of generalized linear model. Journal of Transportation Technologies, 7(03), 318.
'''
def quad_occ_reward(occupancy):
    if 0 < occupancy <= 12:
        return ((0.5 * occupancy) + 6) / 12
    elif 12 < occupancy < 80:
        return ((occupancy-80)**2/68**2)
    else:
        return 0

# Minimize emissions by providing higher rewards for lower emission levels.
def emission_reward(avg_emission):
    """
    Calculate the reward based on CO2 emissions.

    Parameters:
    - avg_emission (CO2_emissions): float, total CO2 emissions at the time of measurement.
    - max_emissions (max_CO2): float, maximum acceptable CO2 emissions for normalization.

    Returns:
    - reward: float, normalized reward between 0 and 1.
    """
    # Ensure max_CO2 is not zero to avoid division by zero
    if max_emissions <= 0:
        raise ValueError("max_CO2 must be greater than zero.")
    
    # Calculate the reward
    reward = max(0, 1 - (avg_emission / max_emissions))
    
    return reward

def enhanced_halts_reward(num_halts, avg_halt_duration, max_halts=10, max_duration=60):
    '''
    For the number of halts, a more detailed approach might consider:
    Frequency and Duration: Penalizing frequent stops and longer durations of halts.
    Impact on Flow: Considering how halts affect overall traffic flow efficiency.
    reward  = 1 - (max halts * max duration / num halts * avg halt duration) 

    - num_halts: Number of stops.
    - avg_halt_duration: Average duration of each halt.
    - max_halts and max_duration: Maximum expected values for normalization.
    '''
    if max_halts <= 0 or max_duration <= 0:
        raise ValueError("max_halts and max_duration must be greater than zero.")
    
    reward = 1 - (num_halts * avg_halt_duration) / (max_halts * max_duration)
    return max(0, reward)

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

# # Minimize total travel time by rewarding faster completion of trips.
# def travel_time_reward(total_travel_time, max_travel_time=3600):
#     # Normalize travel time to a range [0, 1]
#     normalized_travel_time = min(total_travel_time / max_travel_time, 1)
#     return 1 - normalized_travel_time  # Higher reward for shorter travel times

def num_halts_reward(num_halts, max_num_halts=10):
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
    halts_reward = num_halts_reward(mean_num_halts)

    # Combine individual rewards into a total reward
    reward = (
        occ_reward * weight_occupancy +
        spd_reward * weight_speed +
        emis_reward * weight_emissions +
        halts_reward * weight_num_halts
    )

    return reward
