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
            0: 30,  # mph
            1: 40,
            2: 50,
            3: 60,
            4: 70
        }
        self.max_step_down = 10  # mph - MUTCD constraint
        
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