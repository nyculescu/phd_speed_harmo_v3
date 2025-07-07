# sar_components/rewards/comfort_rewards.py
"""Passenger comfort and ride quality focused reward functions"""

import numpy as np
from typing import Dict, Any
from collections import deque

from core.sar_framework import RewardFunction, TrafficMetrics


class PassengerComfortReward(RewardFunction):
    """
    Reward function focused on passenger comfort.
    
    Considers:
    - Acceleration/deceleration limits
    - Jerk (rate of acceleration change)
    - Speed limit change frequency
    - Predictability of driving conditions
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.comfort_accel_limit = 2.0  # m/s² - comfortable acceleration
        self.comfort_jerk_limit = 0.9   # m/s³ - comfortable jerk
        self.speed_history = deque(maxlen=10)
        self.accel_history = deque(maxlen=10)
        self.speed_limit_history = deque(maxlen=20)
        
    def calculate(self, metrics: TrafficMetrics,
                  action_penalty: float = 0.0,
                  collision_penalty: float = 0.0) -> float:
        
        self.speed_history.append(metrics.avg_speed_before)
        self.speed_limit_history.append(metrics.current_speed_limit)
        
        comfort_score = 0
        
        # Acceleration comfort
        if len(self.speed_history) >= 2:
            # Calculate acceleration
            accel = (self.speed_history[-1] - self.speed_history[-2]) / 60  # m/s²
            self.accel_history.append(accel)
            
            # Penalize uncomfortable acceleration
            if abs(accel) > self.comfort_accel_limit:
                accel_penalty = -0.2 * (abs(accel) / self.comfort_accel_limit - 1)
            else:
                accel_penalty = 0.1  # Bonus for smooth acceleration
            
            comfort_score += accel_penalty
        
        # Jerk comfort (change in acceleration)
        if len(self.accel_history) >= 2:
            jerk = (self.accel_history[-1] - self.accel_history[-2]) / 60  # m/s³
            
            if abs(jerk) > self.comfort_jerk_limit:
                jerk_penalty = -0.15 * (abs(jerk) / self.comfort_jerk_limit - 1)
            else:
                jerk_penalty = 0.05
            
            comfort_score += jerk_penalty
        
        # Speed limit change frequency penalty
        if len(self.speed_limit_history) >= 5:
            recent_limits = list(self.speed_limit_history)[-5:]
            changes = sum(1 for i in range(1, len(recent_limits)) 
                         if recent_limits[i] != recent_limits[i-1])
            if changes > 2:
                comfort_score -= 0.1 * (changes - 2)
        
        # Predictability bonus (low variance in conditions)
        if len(self.speed_history) >= 5:
            speed_variance = np.var(list(self.speed_history)[-5:])
            if speed_variance < 4:  # m²/s²
                comfort_score += 0.1
        
        # Flow component (still want reasonable throughput)
        flow_component = 0.3 * min(metrics.flow_smoothed / self.max_flow, 1.0)
        
        # Extreme condition penalties
        if metrics.avg_speed_before < 10 / 3.6:  # Very slow
            comfort_score -= 0.2  # Frustrating for passengers
        
        return float(comfort_score + flow_component + action_penalty + collision_penalty)


class SmoothFlowReward(RewardFunction):
    """
    Reward function for creating smooth, predictable traffic flow.
    
    Focuses on:
    - Minimizing speed variance across time and space
    - Maintaining consistent gaps
    - Reducing wave propagation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.flow_history = deque(maxlen=10)
        self.occupancy_history = deque(maxlen=10)
        
    def calculate(self, metrics: TrafficMetrics,
                  action_penalty: float = 0.0,
                  collision_penalty: float = 0.0) -> float:
        
        self.flow_history.append(metrics.flow_downstream)
        self.occupancy_history.append(metrics.occupancy_upstream)
        
        # Flow stability
        if len(self.flow_history) >= 5:
            flow_std = np.std(list(self.flow_history))
            flow_stability = 0.3 * np.exp(-flow_std / 500)  # Exponential decay
        else:
            flow_stability = 0
        
        # Speed harmonization
        if len(metrics.speed_history) >= 5:
            speed_cv = np.std(list(metrics.speed_history)) / (np.mean(list(metrics.speed_history)) + 0.1)
            speed_harmony = 0.3 * (1 - min(speed_cv, 1.0))
        else:
            speed_harmony = 0
        
        # Occupancy stability (reduces waves)
        if len(self.occupancy_history) >= 5:
            occ_std = np.std(list(self.occupancy_history))
            occ_stability = 0.2 * np.exp(-occ_std / 20)
        else:
            occ_stability = 0
        
        # Throughput maintenance
        throughput = 0.2 * min(metrics.flow_smoothed / self.max_flow, 1.0)
        
        # Wave dampening bonus
        if (len(self.occupancy_history) >= 3 and 
            all(15 < o < 35 for o in list(self.occupancy_history)[-3:])):
            wave_bonus = 0.1  # Maintaining optimal density range
        else:
            wave_bonus = 0
        
        # Queue prevention
        queue_penalty = -0.1 * (metrics.queue_length_upstream / self.max_queue_length) ** 2
        
        return float(flow_stability + speed_harmony + occ_stability + 
                    throughput + wave_bonus + queue_penalty + 
                    action_penalty + collision_penalty)