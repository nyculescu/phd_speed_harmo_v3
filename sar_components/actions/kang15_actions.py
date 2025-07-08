# sar_components/actions/dqnvsl_actions.py
"""DQN-VSL action strategy from Kang et al. (2024)"""

import numpy as np
import gymnasium as gym
from typing import Tuple, Dict, Any
from collections import deque
import logging

from core.sar_framework import ActionStrategy

logger = logging.getLogger(__name__)


class KANG15Action(ActionStrategy):
    """
    DQN-VSL action strategy from Kang et al. (2024).
    
    Actions represent speed limit changes: {-20, -10, 0, +10, +20} km/h
    Resulting speed limits are constrained to: {60, 70, 80, 90, 100} km/h
    
    Constraints:
    - Maximum speed: 100 km/h (free flow speed)
    - Minimum speed: 60 km/h (acceptable traffic flow speed)
    - Maximum change per step: 20 km/h
    """
    
    def _setup(self):
        """Initialize the DQN-VSL action strategy."""
        # Action set: speed changes in km/h
        self.speed_changes = [-20, -10, 0, 10, 20]
        
        # Speed limit constraints
        self.vsl_min = 60  # km/h
        self.vsl_max = 100  # km/h
        self.v_diff = 20   # km/h - maximum change per step
        
        # Track recent changes for analysis
        self.action_history = deque(maxlen=10)
        
        logger.info(f"Initialized DQN-VSL action strategy with changes: {self.speed_changes}")
    
    def get_action_space(self) -> gym.spaces.Space:
        """Return the action space (5 discrete actions)."""
        return gym.spaces.Discrete(len(self.speed_changes))
    
    def apply_action(self, action: int, current_speed_limit: float) -> Tuple[float, float]:
        """
        Apply the speed change action and return new speed limit with penalty.
        
        Args:
            action: Action index (0-4)
            current_speed_limit: Current speed limit in km/h
            
        Returns:
            Tuple of (new_speed_limit, action_penalty)
        """
        if action >= len(self.speed_changes):
            logger.warning(f"Invalid action {action}, maintaining current speed limit")
            return current_speed_limit, -1.0
        
        # Get the speed change
        speed_change = self.speed_changes[action]
        
        # Calculate proposed speed limit
        proposed_speed = current_speed_limit + speed_change
        
        # Apply constraints
        # 1. Check maximum change constraint
        if abs(speed_change) > self.v_diff:
            logger.warning(f"Speed change {speed_change} exceeds maximum difference {self.v_diff}")
            penalty = -0.5
        else:
            penalty = 0.0
        
        # 2. Apply min/max bounds
        new_speed_limit = np.clip(proposed_speed, self.vsl_min, self.vsl_max)
        
        # Additional penalty for hitting bounds
        if proposed_speed < self.vsl_min or proposed_speed > self.vsl_max:
            penalty -= 0.2
            logger.debug(f"Speed limit {proposed_speed} hit bounds [{self.vsl_min}, {self.vsl_max}]")
        
        # Track action for analysis
        self.action_history.append({
            'action': action,
            'change': speed_change,
            'from': current_speed_limit,
            'to': new_speed_limit,
            'penalty': penalty
        })
        
        # Small penalty for frequent changes (optional stability incentive)
        if len(self.action_history) >= 3:
            recent_changes = [h['change'] for h in list(self.action_history)[-3:]]
            if all(c != 0 for c in recent_changes):
                penalty -= 0.1  # Discourage too frequent changes
        
        logger.debug(f"Applied action {action}: {current_speed_limit} -> {new_speed_limit} km/h (penalty: {penalty})")
        
        return new_speed_limit, penalty
    
    def get_valid_actions_mask(self, current_speed_limit: float) -> np.ndarray:
        """
        Get a mask of valid actions given current speed limit.
        
        This is optional and can be used for action masking in training.
        
        Args:
            current_speed_limit: Current speed limit in km/h
            
        Returns:
            Boolean mask of valid actions
        """
        mask = np.ones(len(self.speed_changes), dtype=bool)
        
        for i, change in enumerate(self.speed_changes):
            new_speed = current_speed_limit + change
            # Mark as invalid if it would exceed bounds
            if new_speed < self.vsl_min or new_speed > self.vsl_max:
                mask[i] = False
        
        return mask