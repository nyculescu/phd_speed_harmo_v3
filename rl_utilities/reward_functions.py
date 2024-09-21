max_emissions_over_time = 1000000 # empirical data after running the simulation at max capacity

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
def quad_occ_reward(occupancy): # FIXME
    if 0 < occupancy <= 12:
        return ((0.5 * occupancy) + 6) / 12
    elif 12 < occupancy < 80:
        return ((occupancy-80)**2/68**2)
    else:
        return 0

# Minimize emissions by providing higher rewards for lower emission levels.
def emission_reward(avg_emission, max_emission):
    # Normalize emissions to a range [0, 1]
    normalized_emission = min(avg_emission / max_emission, 1)
    return 1 - normalized_emission  # Higher reward for lower emissions

# Minimize total travel time by rewarding faster completion of trips.
def travel_time_reward(total_travel_time, max_travel_time=3600):
    # Normalize travel time to a range [0, 1]
    normalized_travel_time = min(total_travel_time / max_travel_time, 1)
    return 1 - normalized_travel_time  # Higher reward for shorter travel times

# Encourage speeds close to a target speed (e.g., optimal flow speed) while penalizing speeds that are too low or too high.
def speed_reward(mean_speed, target_speed_mps):
    # Reward is maximum when mean speed is at target speed
    if mean_speed < 0:
        return 0
    elif mean_speed <= target_speed_mps:
        return mean_speed / target_speed_mps
    else:
        # Penalize speeds significantly higher than the target to ensure safety
        return max(0, 1 - ((mean_speed - target_speed_mps) / target_speed_mps))

def wait_time_penalty(total_waiting_time):
    # Penalize both total waiting time and number of stops
    return (-1) * (total_waiting_time / 3600)

def complex_reward(target_speed_mps, mean_speeds, occupancy, avg_emission, total_waiting_time, total_travel_time):
    weight_speed = 0.225
    weight_occupancy = 0.225
    weight_emissions = 0.225
    weight_travel_time = 0.225
    waiting_penalty_weight = 0.1

    # Calculate rewards for each parameter
    spd_reward = speed_reward(mean_speeds, target_speed_mps)
    emis_reward = emission_reward(avg_emission, max_emissions_over_time)
    twt_penalty = wait_time_penalty(total_waiting_time)
    occ_reward = quad_occ_reward(occupancy)
    tts_reward = travel_time_reward(total_travel_time)

    # Combine individual rewards into a total reward
    reward = (
        occ_reward * weight_occupancy +
        spd_reward * weight_speed +
        emis_reward * weight_emissions +
        tts_reward * weight_travel_time +
        twt_penalty * waiting_penalty_weight
    )
    
    # Normalize weights so they sum to 1 if needed
    total_weight = weight_occupancy + weight_speed + weight_emissions + weight_travel_time
    reward /= total_weight

    return reward
