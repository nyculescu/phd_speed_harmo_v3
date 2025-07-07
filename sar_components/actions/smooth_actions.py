# sar_components/actions/smooth_actions.py
"""Smooth action strategies for passenger comfort and stability"""

import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any
from collections import deque

from core.sar_framework import ActionStrategy


class GradualChangeAction(ActionStrategy):
    """
    Ensures gradual speed limit changes for passenger comfort.
    
    Limits the rate of change and penalizes frequent adjustments.
    """
    
    def _setup(self):
        self.max_change_per_step = 10  # km/h
        self.change_history = deque(maxlen=10)
        self.last_change_time = 0
        self.min_time_between_changes = 3  # steps
        self.speed_changes = [-10, -5, -2, 0, 2, 5, 10]
        
    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(len(self.speed_changes))
    
    def apply_action(self, action: int, current_speed_limit: float) -> Tuple[float, float]:
        change = self.speed_changes[action]
        self.change_history.append(change)
        
        # Check time since last change
        time_since_change = getattr(self, 'current_step', 0) - self.last_change_time
        
        penalty = 0
        
        # Penalize frequent changes
        if change != 0 and time_since_change < self.min_time_between_changes:
            penalty -= 0.3
        
        # Penalize oscillations
        if len(self.change_history) >= 3:
            recent_changes = list(self.change_history)[-3:]
            if recent_changes[0] * recent_changes[2] < 0 and recent_changes[0] != 0:
                # Oscillating pattern detected
                penalty -= 0.5
        
        # Apply change with limits
        new_speed = current_speed_limit + change
        new_speed = np.clip(new_speed, 50, 130)
        
        # Update tracking
        if change != 0:
            self.last_change_time = getattr(self, 'current_step', 0)
        
        # Comfort penalty for large changes
        comfort_penalty = -0.05 * (abs(change) / self.max_change_per_step) ** 2
        
        return new_speed, penalty + comfort_penalty


class MomentumBasedAction(ActionStrategy):
    """
    Uses momentum concept to smooth speed transitions.
    
    Previous changes influence current possibilities.
    """
    
    def _setup(self):
        self.momentum = 0
        self.momentum_decay = 0.8
        self.base_actions = [-15, -10, -5, 0, 5, 10, 15]
        self.momentum_history = deque(maxlen=5)
        
    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(len(self.base_actions))
    
    def update_momentum(self, change: float):
        """Update momentum based on recent changes"""
        self.momentum = self.momentum * self.momentum_decay + change * 0.3
        self.momentum = np.clip(self.momentum, -10, 10)
        self.momentum_history.append(self.momentum)
    
    def apply_action(self, action: int, current_speed_limit: float) -> Tuple[float, float]:
        base_change = self.base_actions[action]
        
        # Modify change based on momentum
        momentum_factor = self.momentum / 10  # Normalize to [-1, 1]
        
        # If momentum and action are in same direction, enhance
        # If opposite, dampen
        if base_change * self.momentum > 0:
            effective_change = base_change * (1 + 0.3 * abs(momentum_factor))
        else:
            effective_change = base_change * (1 - 0.3 * abs(momentum_factor))
        
        # Apply limits
        effective_change = np.clip(effective_change, -20, 20)
        new_speed = np.clip(current_speed_limit + effective_change, 50, 130)
        
        # Update momentum
        actual_change = new_speed - current_speed_limit
        self.update_momentum(actual_change)
        
        # Penalties
        penalty = 0
        
        # Penalty for fighting momentum (sudden direction changes)
        if base_change * self.momentum < -50:  # Strong opposition
            penalty -= 0.3
        
        # Reward smooth transitions
        if len(self.momentum_history) >= 3:
            momentum_variance = np.var(list(self.momentum_history))
            if momentum_variance < 5:
                penalty += 0.1  # Reward consistent momentum
        
        return new_speed, penalty