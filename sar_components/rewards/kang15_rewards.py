# sar_components/rewards/dqnvsl_rewards.py
"""DQN-VSL reward function from Kang et al. (2024)"""

import numpy as np
from typing import Dict, Any
import logging

from core.sar_framework import RewardFunction, TrafficMetrics

logger = logging.getLogger(__name__)


class KANG15Reward(RewardFunction):
    """
    DQN-VSL reward function from Kang et al. (2024).
    
    The reward combines safety and efficiency components:
    - Safety: Based on Time-to-Collision (TTC) with forward-looking component
    - Efficiency: Based on speed improvement ratio
    
    Formula:
    R_i^t = 100 * (R_safety,i^t + R_efficiency,i^t)
    
    Where:
    - r_safety,i^t = 1 - exp(-TTC_i^t)
    - R_safety,i^t = r_safety,i × 0.75 + r_safety,i+1 × 0.25
    - R_efficiency,i^t = (Speed_i^t - Speed_i^(t-1)) / Speed_i^(t-1)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Weights for current and next section safety
        self.current_weight = 0.75
        self.next_weight = 0.25
        
        # Scaling factor
        self.reward_scale = 100.0
        
        # Store previous speed for efficiency calculation
        self.previous_speed = None
        
        # TTC threshold for safety calculation
        self.ttc_threshold = 4.0  # seconds
        
        # Minimum speed to avoid division by zero
        self.min_speed_threshold = 1.0  # m/s
        
        logger.info("Initialized DQN-VSL reward function")
    
    def calculate(self, metrics: TrafficMetrics, 
                  action_penalty: float = 0.0,
                  collision_penalty: float = 0.0) -> float:
        """
        Calculate the DQN-VSL reward based on safety and efficiency.
        
        Args:
            metrics: Current traffic metrics
            action_penalty: Penalty from action strategy
            collision_penalty: Penalty for collisions
            
        Returns:
            Total reward value
        """
        # Calculate safety reward components
        r_safety_current = self._calculate_safety_component(metrics)
        r_safety_next = self._calculate_safety_component_next(metrics)
        
        # Weighted safety reward
        R_safety = self.current_weight * r_safety_current + self.next_weight * r_safety_next
        
        # Calculate efficiency reward
        R_efficiency = self._calculate_efficiency_component(metrics)
        
        # Total reward (scaled by 100 as in paper)
        total_reward = self.reward_scale * (R_safety + R_efficiency)
        
        # Add penalties
        total_reward += action_penalty + collision_penalty
        
        # Log components for debugging
        logger.debug(f"Reward components - Safety: {R_safety:.3f}, Efficiency: {R_efficiency:.3f}, "
                    f"Action penalty: {action_penalty:.3f}, Total: {total_reward:.3f}")
        
        # Update previous speed for next calculation
        self.previous_speed = metrics.avg_speed_before
        
        return float(total_reward)
    
    def _calculate_safety_component(self, metrics: TrafficMetrics) -> float:
        """
        Calculate safety component based on TTC.
        
        r_safety = 1 - exp(-TTC)
        """
        # Approximate TTC calculation based on available metrics
        ttc = self._estimate_ttc(metrics)
        
        # Safety reward formula from paper
        r_safety = 1.0 - np.exp(-ttc)
        
        return float(r_safety)
    
    def _calculate_safety_component_next(self, metrics: TrafficMetrics) -> float:
        """
        Calculate safety component for next section.
        
        Uses upstream metrics as proxy for next section.
        """
        # Create approximate metrics for next section
        next_ttc = self._estimate_ttc_next(metrics)
        
        # Safety reward for next section
        r_safety_next = 1.0 - np.exp(-next_ttc)
        
        return float(r_safety_next)
    
    def _calculate_efficiency_component(self, metrics: TrafficMetrics) -> float:
        """
        Calculate efficiency component based on speed improvement.
        
        R_efficiency = (Speed_t - Speed_{t-1}) / Speed_{t-1}
        """
        if self.previous_speed is None:
            # First step, no previous speed available
            return 0.0
        
        # Ensure we don't divide by zero
        if self.previous_speed < self.min_speed_threshold:
            return 0.0
        
        # Calculate speed improvement ratio
        speed_improvement = (metrics.avg_speed_before - self.previous_speed) / self.previous_speed
        
        # Clip to reasonable range to avoid extreme values
        R_efficiency = np.clip(speed_improvement, -0.5, 0.5)
        
        return float(R_efficiency)
    
    def _estimate_ttc(self, metrics: TrafficMetrics) -> float:
        """
        Estimate TTC from available metrics.
        
        This is a simplified estimation since individual vehicle TTCs
        are not directly available in the TrafficMetrics.
        """
        # Higher TTC (safer) when:
        # - Lower occupancy
        # - Higher speed
        # - Lower queue length
        
        if metrics.occupancy_upstream > 80:  # High congestion
            # Very low TTC in congestion
            ttc = 0.5
        elif metrics.queue_length_upstream > 100:  # Significant queue
            # Low TTC with queuing
            ttc = 1.0
        elif metrics.avg_speed_before < 10:  # Very slow speed
            # Low TTC in slow traffic
            ttc = 1.5
        else:
            # Estimate based on speed and occupancy
            # Higher speed and lower occupancy = higher TTC
            speed_factor = min(metrics.avg_speed_before / 30.0, 1.0)  # Normalize to ~100 km/h
            occupancy_factor = 1.0 - (metrics.occupancy_upstream / 100.0)
            ttc = 2.0 + 2.0 * speed_factor * occupancy_factor
        
        # Cap at threshold
        return min(ttc, self.ttc_threshold)
    
    def _estimate_ttc_next(self, metrics: TrafficMetrics) -> float:
        """
        Estimate TTC for next section using upstream metrics.
        """
        # Use upstream metrics as proxy
        # Typically upstream has better conditions
        if metrics.upstream_speed > metrics.avg_speed_before:
            # Better conditions upstream
            ttc_next = self._estimate_ttc(metrics) * 1.2
        else:
            # Similar or worse conditions
            ttc_next = self._estimate_ttc(metrics) * 0.9
        
        return min(ttc_next, self.ttc_threshold)
    
    def reset_metrics(self):
        """Reset internal state for new episode."""
        self.previous_speed = None
        logger.debug("Reset DQN-VSL reward metrics")