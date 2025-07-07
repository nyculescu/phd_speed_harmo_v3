# sar_components/states/marvel_states.py
"""MARVEL-specific state representations"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any

from core.sar_framework import StateRepresentation, TrafficMetrics


class MARVELState(StateRepresentation):
    """
    MARVEL state representation that includes downstream agent's action.
    
    State tuple: ⟨a^(i-1)_t, ν^i_t, o^i_t, ν^(i+1)_t, o^(i+1)_t⟩
    - a^(i-1)_t: action from preceding (downstream) agent
    - ν^i_t, o^i_t: speed and occupancy from collocated sensor
    - ν^(i+1)_t, o^(i+1)_t: speed and occupancy from upstream sensor
    """
    
    def _setup(self):
        self.num_features = 5
        self.max_speed_mps = 112.7 / 3.6  # 70 mph in m/s (converted to km/h first)
        self.max_occupancy = 100.0
        self.downstream_action = 112.7  # Default 70 mph in km/h
        
    def get_observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(
            low=np.array([48.3, 0, 0, 0, 0], dtype=np.float64),  # 30 mph in km/h
            high=np.array([112.7, self.max_speed_mps, self.max_occupancy,  # 70 mph in km/h
                        self.max_speed_mps, self.max_occupancy], dtype=np.float64),
            shape=(self.num_features,),
            dtype=np.float64
        )
    
    def set_downstream_action(self, action: float):
        """Set the action from downstream agent for spatially sequential decision-making"""
        self.downstream_action = action
    
    def build_state(self, metrics: TrafficMetrics) -> np.ndarray:       
        return np.array([
            self.downstream_action,
            metrics.avg_speed_before,
            metrics.occupancy_upstream,
            metrics.upstream_speed,
            metrics.upstream_occupancy
        ], dtype=np.float64)
    
    def preprocess_state(self, raw_state: np.ndarray) -> np.ndarray:
        # Normalize each component
        normalized = np.zeros_like(raw_state)
        normalized[0] = (raw_state[0] - 48.3) / 64.4  # Action: 30-70 mph range in km/h
        normalized[1] = raw_state[1] / self.max_speed_mps  # Speed
        normalized[2] = raw_state[2] / self.max_occupancy  # Occupancy
        normalized[3] = raw_state[3] / self.max_speed_mps  # Upstream speed
        normalized[4] = raw_state[4] / self.max_occupancy  # Upstream occupancy
        
        return normalized