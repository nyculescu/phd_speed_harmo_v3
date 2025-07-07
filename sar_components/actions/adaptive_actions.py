# sar_components/actions/adaptive_actions.py
"""Adaptive action strategies that respond to traffic conditions"""

import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any
from collections import deque

from core.sar_framework import ActionStrategy


class TrafficAdaptiveAction(ActionStrategy):
    """
    Adapts available actions based on current traffic conditions.
    
    In free flow: Allow higher speeds
    In congestion: Restrict to lower speeds for stability
    """
    
    def _setup(self):
        self.base_speeds = {
            'free_flow': [90, 100, 110, 120, 130],
            'moderate': [70, 80, 90, 100, 110],
            'congested': [50, 60, 70, 80, 90]
        }
        self.occupancy_history = deque(maxlen=5)
        self.current_regime = 'free_flow'
        
    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(5)  # Always 5 actions, but they change
    
    def determine_traffic_regime(self, occupancy: float) -> str:
        """Determine traffic regime based on occupancy"""
        self.occupancy_history.append(occupancy)
        avg_occupancy = np.mean(self.occupancy_history) if self.occupancy_history else occupancy
        
        if avg_occupancy < 25:
            return 'free_flow'
        elif avg_occupancy < 45:
            return 'moderate'
        else:
            return 'congested'
    
    def apply_action(self, action: int, current_speed_limit: float) -> Tuple[float, float]:
        # Get current occupancy from environment (would need to be passed in)
        # For now, use a placeholder
        current_occupancy = getattr(self, 'last_occupancy', 30)
        
        # Update traffic regime
        self.current_regime = self.determine_traffic_regime(current_occupancy)
        
        # Get appropriate speed set
        available_speeds = self.base_speeds[self.current_regime]
        
        if action >= len(available_speeds):
            return current_speed_limit, -1.0
        
        new_speed = available_speeds[action]
        
        # Penalty for large changes
        change = abs(new_speed - current_speed_limit)
        if change > 20:
            penalty = -0.3
        elif change > 10:
            penalty = -0.1
        else:
            penalty = 0
        
        # Bonus for appropriate response
        if self.current_regime == 'congested' and new_speed < current_speed_limit:
            penalty += 0.1  # Reward speed reduction in congestion
        
        return new_speed, penalty


class OccupancyBasedAction(ActionStrategy):
    """
    Direct mapping from occupancy levels to speed recommendations.
    
    Uses a continuous function to determine speed based on occupancy.
    """
    
    def _setup(self):
        # Define occupancy thresholds and corresponding speeds
        self.occupancy_thresholds = [0, 15, 25, 35, 45, 60, 80, 100]
        self.speed_targets = [130, 120, 110, 100, 90, 70, 60, 50]
        self.action_offsets = [-10, -5, 0, 5, 10]  # Modifications to target
        
    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(len(self.action_offsets))
    
    def get_target_speed(self, occupancy: float) -> float:
        """Interpolate target speed based on occupancy"""
        # Linear interpolation between thresholds
        for i in range(len(self.occupancy_thresholds) - 1):
            if self.occupancy_thresholds[i] <= occupancy < self.occupancy_thresholds[i + 1]:
                # Interpolate
                ratio = (occupancy - self.occupancy_thresholds[i]) / \
                       (self.occupancy_thresholds[i + 1] - self.occupancy_thresholds[i])
                target = self.speed_targets[i] + ratio * (self.speed_targets[i + 1] - self.speed_targets[i])
                return target
        
        return self.speed_targets[-1]  # Maximum congestion
    
    def apply_action(self, action: int, current_speed_limit: float) -> Tuple[float, float]:
        # Get current occupancy (would need to be passed in)
        current_occupancy = getattr(self, 'last_occupancy', 30)
        
        # Get target speed for current occupancy
        target_speed = self.get_target_speed(current_occupancy)
        
        # Apply action offset
        offset = self.action_offsets[action]
        new_speed = np.clip(target_speed + offset, 50, 130)
        
        # Penalty based on deviation from target
        deviation_penalty = -0.1 * abs(new_speed - target_speed) / 20
        
        # Penalty for not following occupancy guidance
        if current_occupancy > 50 and new_speed > 90:
            deviation_penalty -= 0.2  # Extra penalty for high speed in congestion
        
        return new_speed, deviation_penalty