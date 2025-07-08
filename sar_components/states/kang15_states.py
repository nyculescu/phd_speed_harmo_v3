# sar_components/states/dqnvsl_states.py
"""DQN-VSL state representation from Kang et al. (2024)"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any
from collections import deque
import logging

from core.sar_framework import StateRepresentation, TrafficMetrics

logger = logging.getLogger(__name__)


class KANG15State(StateRepresentation):
    """
    DQN-VSL state representation from Kang et al. (2024).
    
    State vector includes 11 features:
    - Desired speed (current, next)
    - Average speed (current, next)
    - Traffic volume (current)
    - Average acceleration (current)
    - Average jerk (current)
    - Average yaw (current)
    - Average exp(-TTC) (current, next)
    - Average CPI (current)
    
    This state representation captures both local traffic conditions
    and forward-looking information from the next section.
    """
    
    def _setup(self):
        """Initialize the DQN-VSL state representation."""
        self.num_features = 11
        
        # Normalization bounds from paper
        self.max_speed_kmh = 130.0  # km/h
        self.max_volume = 10000.0  # vehicles/hour
        self.max_acceleration = 4.0  # m/s²
        self.max_jerk = 2.0  # m/s³
        self.max_yaw = np.pi  # radians
        
        # TTC parameters
        self.ttc_threshold = 4.0  # seconds
        
        # CPI bounds
        self.max_cpi = 1.0
        
        logger.info("Initialized DQN-VSL state representation with 11 features")
    
    def get_observation_space(self) -> gym.spaces.Space:
        """Return the observation space for DQN-VSL representation."""
        return gym.spaces.Box(
            low=np.zeros(self.num_features, dtype=np.float64),
            high=np.ones(self.num_features, dtype=np.float64),
            shape=(self.num_features,),
            dtype=np.float64
        )
    
    def build_state(self, metrics: TrafficMetrics) -> np.ndarray:
        """
        Build the raw state vector from traffic metrics.
        
        Args:
            metrics: Current traffic measurements
            
        Returns:
            Raw state vector with 11 features
        """
        # Calculate average exponential of negative TTC
        avg_exp_neg_ttc = self._calculate_avg_exp_neg_ttc(metrics)
        
        # Get downstream values (using upstream as proxy for next section)
        next_speed = metrics.upstream_speed if metrics.upstream_speed > 0 else metrics.avg_speed_before
        next_exp_neg_ttc = avg_exp_neg_ttc * 0.8  # Approximate next section
        
        # Build state vector
        state = np.array([
            metrics.current_speed_limit,      # Desired speed (current)
            metrics.downstream_speed_limit,   # Desired speed (next)
            metrics.avg_speed_before * 3.6,   # Average speed in km/h (current)
            next_speed * 3.6,                 # Average speed in km/h (next)
            metrics.flow_upstream,            # Traffic volume (current)
            self._calculate_avg_acceleration(metrics),  # Average acceleration
            self._calculate_avg_jerk(metrics),         # Average jerk
            self._calculate_avg_yaw(metrics),          # Average yaw
            avg_exp_neg_ttc,                          # Average exp(-TTC) (current)
            next_exp_neg_ttc,                         # Average exp(-TTC) (next)
            self._calculate_avg_cpi(metrics)          # Average CPI
        ], dtype=np.float64)
        
        return state
    
    def preprocess_state(self, raw_state: np.ndarray) -> np.ndarray:
        """
        Normalize the raw state for the neural network.
        
        Args:
            raw_state: Raw state vector
            
        Returns:
            Normalized state vector with values in [0, 1]
        """
        if len(raw_state) != self.num_features:
            logger.error(f"State dimension mismatch: {len(raw_state)} vs {self.num_features}")
            return np.zeros(self.num_features, dtype=np.float64)
        
        normalized = np.zeros_like(raw_state)
        
        # Normalize each feature
        normalized[0] = np.clip(raw_state[0], 60, 130) / self.max_speed_kmh  # Desired speed (current)
        normalized[1] = np.clip(raw_state[1], 60, 130) / self.max_speed_kmh  # Desired speed (next)
        normalized[2] = np.clip(raw_state[2], 0, self.max_speed_kmh) / self.max_speed_kmh  # Avg speed (current)
        normalized[3] = np.clip(raw_state[3], 0, self.max_speed_kmh) / self.max_speed_kmh  # Avg speed (next)
        normalized[4] = np.clip(raw_state[4], 0, self.max_volume) / self.max_volume  # Volume
        normalized[5] = np.clip(raw_state[5], -self.max_acceleration, self.max_acceleration) / (2 * self.max_acceleration) + 0.5  # Acceleration
        normalized[6] = np.clip(raw_state[6], -self.max_jerk, self.max_jerk) / (2 * self.max_jerk) + 0.5  # Jerk
        normalized[7] = np.clip(raw_state[7], -self.max_yaw, self.max_yaw) / (2 * self.max_yaw) + 0.5  # Yaw
        normalized[8] = np.clip(raw_state[8], 0, 1)  # Exp(-TTC) current (already in [0,1])
        normalized[9] = np.clip(raw_state[9], 0, 1)  # Exp(-TTC) next (already in [0,1])
        normalized[10] = np.clip(raw_state[10], 0, self.max_cpi) / self.max_cpi  # CPI
        
        return normalized
    
    def _calculate_avg_exp_neg_ttc(self, metrics: TrafficMetrics) -> float:
        """Calculate average exponential of negative TTC."""
        # This is a simplified calculation based on available metrics
        # In the actual implementation, individual vehicle TTCs would be used
        if metrics.avg_speed_before > 0 and metrics.queue_length_upstream > 0:
            # Approximate TTC based on queue and speed
            approx_ttc = metrics.queue_length_upstream / (metrics.avg_speed_before * 10)
            return 1 - np.exp(-min(approx_ttc, self.ttc_threshold))
        return 0.0
    
    def _calculate_avg_acceleration(self, metrics: TrafficMetrics) -> float:
        """Calculate average acceleration from speed history."""
        if len(metrics.speed_history) >= 2:
            speeds = list(metrics.speed_history)
            # Calculate acceleration between consecutive measurements
            accel = (speeds[-1] - speeds[-2]) / 60.0  # Assuming 60s intervals
            return float(accel)
        return 0.0
    
    def _calculate_avg_jerk(self, metrics: TrafficMetrics) -> float:
        """Calculate average jerk (rate of acceleration change)."""
        if len(metrics.speed_history) >= 3:
            speeds = list(metrics.speed_history)
            # Calculate accelerations
            accel1 = (speeds[-2] - speeds[-3]) / 60.0
            accel2 = (speeds[-1] - speeds[-2]) / 60.0
            # Calculate jerk
            jerk = (accel2 - accel1) / 60.0
            return float(jerk)
        return 0.0
    
    def _calculate_avg_yaw(self, metrics: TrafficMetrics) -> float:
        """Calculate average yaw (simplified as speed variance indicator)."""
        # In a real implementation, this would use vehicle trajectory data
        # Here we approximate using speed variance as a proxy
        if len(metrics.speed_history) >= 3:
            speed_var = np.var(list(metrics.speed_history))
            # Convert variance to a yaw-like metric
            return float(np.tanh(speed_var / 100.0) * 0.1)  # Small yaw values
        return 0.0
    
    def _calculate_avg_cpi(self, metrics: TrafficMetrics) -> float:
        """Calculate average Crash Potential Index."""
        # Simplified CPI calculation based on available metrics
        # Higher occupancy and lower speed indicate higher crash potential
        if metrics.occupancy_upstream > 50 and metrics.avg_speed_before < 20:
            cpi = min((metrics.occupancy_upstream / 100.0) * (1 - metrics.avg_speed_before / 30.0), 1.0)
        else:
            cpi = 0.0
        return float(cpi)