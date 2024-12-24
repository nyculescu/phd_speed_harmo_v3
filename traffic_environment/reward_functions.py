import numpy as np
import logging
from traffic_environment.road import *

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

max_emissions = 250000 # empirical data after running the simulation at max capacity
max_occupancy = 50000 # empirical data after running the simulation at max capacity

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