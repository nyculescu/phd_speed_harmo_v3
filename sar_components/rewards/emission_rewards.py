# sar_components/rewards/emission_rewards.py
"""Emission and environmental focused reward functions"""

import numpy as np
from typing import Dict, Any

from core.sar_framework import RewardFunction, TrafficMetrics


class EmissionReward(RewardFunction):
    """
    Reward function focused on minimizing vehicle emissions.
    
    Based on:
    - Optimal speed range for emissions (80-100 km/h)
    - Minimizing acceleration/deceleration cycles
    - Reducing stop-and-go traffic
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.optimal_speed_range = (80 / 3.6, 100 / 3.6)  # m/s
        self.speed_history = []
        
    def calculate(self, metrics: TrafficMetrics,
                  action_penalty: float = 0.0,
                  collision_penalty: float = 0.0) -> float:
        
        # Speed efficiency for emissions
        speed_efficiency = 0
        if self.optimal_speed_range[0] <= metrics.avg_speed_before <= self.optimal_speed_range[1]:
            speed_efficiency = 0.4
        else:
            # Penalty increases with distance from optimal range
            if metrics.avg_speed_before < self.optimal_speed_range[0]:
                distance = self.optimal_speed_range[0] - metrics.avg_speed_before
            else:
                distance = metrics.avg_speed_before - self.optimal_speed_range[1]
            speed_efficiency = -0.2 * (distance / 10) ** 2
        
        # Stop-and-go penalty (high emissions)
        self.speed_history.append(metrics.avg_speed_before)
        if len(self.speed_history) > 5:
            self.speed_history.pop(0)
            speed_variance = np.var(self.speed_history)
            stop_go_penalty = -0.3 * (speed_variance / 100)
        else:
            stop_go_penalty = 0
        
        # Flow component (maintain reasonable throughput)
        flow_component = 0.2 * min(metrics.flow_smoothed / self.max_flow, 1.0)
        
        # Queue penalty (idling vehicles produce emissions)
        if metrics.queue_length_upstream > 0:
            idle_penalty = -0.2 * (metrics.queue_length_upstream / self.max_queue_length)
        else:
            idle_penalty = 0
        
        # Low speed penalty (inefficient combustion)
        if metrics.avg_speed_before < 50 / 3.6:  # Below 50 km/h
            low_speed_penalty = -0.1 * (1 - metrics.avg_speed_before / (50 / 3.6))
        else:
            low_speed_penalty = 0
        
        return float(speed_efficiency + stop_go_penalty + flow_component + 
                    idle_penalty + low_speed_penalty + action_penalty + collision_penalty)


class FuelEfficiencyReward(RewardFunction):
    """
    Reward function focused on minimizing fuel consumption.
    
    Considers:
    - Steady-state cruising benefits
    - Acceleration fuel costs
    - Optimal speed for fuel economy (90 km/h)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.optimal_fuel_speed = 90 / 3.6  # m/s
        self.previous_speed = None
        self.acceleration_history = []
        
    def calculate(self, metrics: TrafficMetrics,
                  action_penalty: float = 0.0,
                  collision_penalty: float = 0.0) -> float:
        
        # Speed optimality for fuel consumption
        speed_deviation = abs(metrics.avg_speed_before - self.optimal_fuel_speed)
        fuel_speed_component = 0.3 * np.exp(-speed_deviation / 10)
        
        # Acceleration penalty
        if self.previous_speed is not None:
            acceleration = (metrics.avg_speed_before - self.previous_speed) / 60  # m/s²
            self.acceleration_history.append(acceleration)
            
            if len(self.acceleration_history) > 10:
                self.acceleration_history.pop(0)
            
            # RMS acceleration (fuel consumption proxy)
            rms_accel = np.sqrt(np.mean(np.square(self.acceleration_history)))
            accel_penalty = -0.3 * min(rms_accel * 10, 1.0)
        else:
            accel_penalty = 0
        
        self.previous_speed = metrics.avg_speed_before
        
        # Steady flow bonus
        if len(metrics.speed_history) > 5:
            speed_std = np.std(list(metrics.speed_history))
            if speed_std < 2:  # Very steady
                steady_bonus = 0.2
            elif speed_std < 5:  # Reasonably steady
                steady_bonus = 0.1
            else:
                steady_bonus = 0
        else:
            steady_bonus = 0
        
        # Throughput component
        throughput = 0.2 * min(metrics.flow_smoothed / self.max_flow, 1.0)
        
        return float(fuel_speed_component + accel_penalty + steady_bonus + 
                    throughput + action_penalty + collision_penalty)