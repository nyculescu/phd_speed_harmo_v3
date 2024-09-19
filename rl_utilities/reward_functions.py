import numpy as np

def normalize(value, min_value, max_value):
    """Normalize a value between min_value and max_value to a range [0, 1]."""
    norm_val = (value - min_value) / (max_value - min_value) if (min_value <= value <= max_value) else 1
    return norm_val

def linear_occ_reward(x):
    if 9 < x < 13:
        return 2

    if 0 < x <= 11:
        return x/11
    elif 11 < x < 100:
        return (100-x)/89
    else:
        return 0

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
def emission_reward(emission_sum, max_emission):
    # Normalize emissions to a range [0, 1]
    normalized_emission = min(emission_sum / max_emission, 1)
    return 1 - normalized_emission  # Higher reward for lower emissions

# Minimize total travel time by rewarding faster completion of trips.
def travel_time_reward(total_travel_time, max_travel_time=3600):
    # Normalize travel time to a range [0, 1]
    normalized_travel_time = min(total_travel_time / max_travel_time, 1)
    return 1 - normalized_travel_time  # Higher reward for shorter travel times

# Encourage speeds close to a target speed (e.g., optimal flow speed) while penalizing speeds that are too low or too high.
def speed_reward(mean_speed, target_speed_mps):
    if mean_speed < 0:
        return 0
    elif mean_speed <= target_speed_mps:
        return (mean_speed / target_speed_mps)  # Increases linearly to 1 at target speed
    else:
        return max(0, 2 - (mean_speed / target_speed_mps))  # Decreases linearly beyond target speed

def complex_reward(target_speed_mps, mean_speeds, occupancy, emission_sum, max_emissions_over_time, total_travel_time):
    weight_speed = 0.3
    weight_occupancy = 0.3
    weight_emissions = 0.2
    weight_travel_time = 0.2

        # Calculate rewards for each parameter
    speed_reward_value = speed_reward(mean_speeds, target_speed_mps)
    emission_reward_value = emission_reward(emission_sum, max_emissions_over_time)
    travel_time_reward_value = travel_time_reward(total_travel_time)
    occupancy_reward = quad_occ_reward(occupancy)

    # Combine individual rewards into a total reward
    reward = (
        occupancy_reward * weight_occupancy +
        speed_reward_value * weight_speed +
        emission_reward_value * weight_emissions +
        travel_time_reward_value * weight_travel_time
    )
    
    # Normalize weights so they sum to 1 if needed
    total_weight = weight_occupancy + weight_speed + weight_emissions + weight_travel_time
    reward /= total_weight

    return reward
