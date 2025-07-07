# sar_components/rewards/marvel_rewards.py
"""MARVEL-specific reward functions"""

import numpy as np
from typing import Dict, Any

from core.sar_framework import RewardFunction, TrafficMetrics


class MARVELReward(RewardFunction):
    """
    MARVEL reward function with three components:
    - Adaptability: Penalize high speed limits in congestion
    - Safety: Enforce maximum step-down constraint
    - Mobility: Encourage higher speeds when possible
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.w1 = 0.2  # Adaptability weight
        self.w2 = 0.3  # Safety weight  
        self.w3 = 0.5  # Mobility weight
        self.congestion_threshold = 56.3 / 3.6  # 35 mph in m/s (converted from km/h)
        self.max_step_down = 16.1  # 10 mph in km/h
        self.downstream_action = 112.7  # Default 70 mph in km/h
        
    def set_downstream_action(self, action: float):
        """Set downstream agent's action for safety reward calculation"""
        self.downstream_action = action
        
    def calculate(self, metrics: TrafficMetrics,
                action_penalty: float = 0.0,
                collision_penalty: float = 0.0) -> float:
        
        # Get current action (speed limit) in km/h
        current_action = metrics.current_speed_limit
        
        # 1. Adaptability reward
        r1 = 0.0
        if metrics.avg_speed_before <= self.congestion_threshold and current_action != 48.3:  # 30 mph
            r1 = -10.0
            
        # 2. Safety reward (max step-down constraint)
        r2 = 0.0
        if hasattr(self, 'agent_index') and self.agent_index == 0:
            # Most downstream agent - no constraint
            r2 = 0.0
        elif self.downstream_action == 48.3 and current_action in [48.3, 64.4]:  # 30, 40 mph
            # Downstream in congestion
            r2 = 0.0
        elif current_action > self.downstream_action + self.max_step_down:
            # Violation
            r2 = -2.0 * (current_action - self.downstream_action) / self.max_step_down
        else:
            # Valid transitions
            if (self.downstream_action == 64.4 and current_action == 80.5) or \
            (self.downstream_action == 80.5 and current_action == 96.6) or \
            (self.downstream_action in [96.6, 112.7] and current_action == 112.7):
                r2 = 2.0
                
        # 3. Mobility reward
        next_speed = metrics.avg_speed_before  # Using current as proxy for next
        v_max = 112.7 / 3.6  # 70 mph in m/s
        v_clip = min(next_speed, v_max)
        r3 = (np.exp(v_clip / v_max) - np.exp(0)) / (np.exp(1) - np.exp(0))
        
        # Total reward
        total_reward = self.w1 * r1 + self.w2 * r2 + self.w3 * r3
        
        return float(total_reward + action_penalty + collision_penalty)