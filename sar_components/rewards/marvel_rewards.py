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

        # Internal metrics tracking
        self._metrics_history = {
            'violations': [],
            'congestion_adaptations': [],
            'reward_components': [],
            'speeds': [],
            'occupancies': []
        }
        
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
        
        self._log_internal_metrics(metrics, r1, r2, r3)

        return float(total_reward + action_penalty + collision_penalty)
    
    def _log_internal_metrics(self, metrics: TrafficMetrics, r1: float, r2: float, r3: float):
        """Internal metrics tracking."""
        # Track violations
        violation = metrics.current_speed_limit > metrics.downstream_speed_limit + self.max_step_down
        self._metrics_history['violations'].append(violation)
        
        # Track congestion adaptations
        in_congestion = metrics.avg_speed_before <= self.congestion_threshold
        adapted = in_congestion and metrics.current_speed_limit == 48.3
        self._metrics_history['congestion_adaptations'].append(adapted)
        
        # Track components
        self._metrics_history['reward_components'].append({
            'r1': r1, 'r2': r2, 'r3': r3
        })
        
        # Track conditions
        self._metrics_history['speeds'].append(metrics.avg_speed_before)
        self._metrics_history['occupancies'].append(metrics.occupancy_upstream)
    
    def get_episode_summary(self) -> Dict[str, float]:
        """Get summary statistics for the episode."""
        if not self._metrics_history['violations']:
            return {}
        
        n_steps = len(self._metrics_history['violations'])
        components = self._metrics_history['reward_components']
        
        return {
            'violation_rate': sum(self._metrics_history['violations']) / n_steps,
            'adaptation_rate': sum(self._metrics_history['congestion_adaptations']) / n_steps,
            'avg_r1': np.mean([c['r1'] for c in components]),
            'avg_r2': np.mean([c['r2'] for c in components]),
            'avg_r3': np.mean([c['r3'] for c in components]),
            'avg_speed': np.mean(self._metrics_history['speeds']),
            'avg_occupancy': np.mean(self._metrics_history['occupancies'])
        }
    
    def reset_metrics(self):
        """Reset internal metrics for new episode."""
        for key in self._metrics_history:
            self._metrics_history[key].clear()