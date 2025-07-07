# sar_components/actions/marvel_actions.py
"""MARVEL-specific action strategies"""

import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any

from core.sar_framework import ActionStrategy


class MARVELSpeedAction(ActionStrategy):
    """
    MARVEL action strategy with specific speed limits: 30, 40, 50, 60, 70 mph
    """
    
    def _setup(self):
        self.speed_actions = {
            0: 48.3,  # 30 mph in km/h
            1: 64.4,  # 40 mph
            2: 80.5,  # 50 mph
            3: 96.6,  # 60 mph
            4: 112.7  # 70 mph
        }
        self.max_step_down = 16.1  # 10 mph in km/h - MUTCD constraint
        
    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(len(self.speed_actions))
    
    def apply_action(self, action: int, current_speed_limit: float) -> Tuple[float, float]:
        proposed_speed_limit = self.speed_actions.get(action, current_speed_limit)
        
        # No explicit penalty here as MARVEL handles it in reward function
        penalty = 0.0
        
        return proposed_speed_limit, penalty
    
    def get_invalid_actions(self, downstream_speed_limit: float) -> list:
        """Get list of invalid actions based on max step-down constraint"""
        invalid_actions = []
        for action, speed in self.speed_actions.items():
            if speed > downstream_speed_limit + self.max_step_down:
                invalid_actions.append(action)
        return invalid_actions
    
    def get_valid_actions(self, current_speed_limit: float, 
                         downstream_speed_limit: float) -> np.ndarray:
        """
        Return mask of valid actions.
        This is optional - the environment doesn't need to know about it.
        """
        mask = np.ones(len(self.speed_actions), dtype=bool)
        
        for action, speed in self.speed_actions.items():
            if speed > downstream_speed_limit + self.max_step_down:
                mask[action] = False
                
        return mask