max_emissions = 2000000 # empirical data after running the simulation at max capacity

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
def emission_reward(avg_emission):
    # Normalize emissions to a range [0, 1]
    normalized_emission = min(avg_emission / max_emissions, 1)
    return 1 - normalized_emission  # Higher reward for lower emissions

# Minimize total travel time by rewarding faster completion of trips.
def travel_time_reward(total_travel_time, max_travel_time=3600):
    # Normalize travel time to a range [0, 1]
    normalized_travel_time = min(total_travel_time / max_travel_time, 1)
    return 1 - normalized_travel_time  # Higher reward for shorter travel times

def num_halts_reward(num_halts, max_num_halts=10):
    normalized_num_halts = min(num_halts / max_num_halts, 1)
    return 1 - normalized_num_halts  # Higher reward for shorter travel times

# Encourage speeds close to a target speed (e.g., optimal flow speed) while penalizing speeds that are too low or too high.
def speed_reward(mean_speed_mps, target_speed_mps):
    # Reward is maximum when mean speed is at target speed
    if mean_speed_mps < 0:
        return 0
    elif mean_speed_mps <= target_speed_mps:
        return mean_speed_mps / target_speed_mps
    else:
        # Penalize speeds significantly higher than the target to ensure safety
        return max(0, 1 - ((mean_speed_mps - target_speed_mps) / target_speed_mps))

def wait_time_penalty(total_waiting_time):
    # Penalize both total waiting time and number of stops
    return (-1) * (total_waiting_time / 3600)

def complex_reward(target_speed_mps, mean_speeds_mps, occupancy, mean_emissions, mean_num_halts):
    weight_speed = 0.25
    weight_occupancy = 0.25
    weight_emissions = 0.25
    weight_num_halts = 0.25

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
