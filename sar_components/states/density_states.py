# sar_components/states/density_states.py
"""Density-focused state representations for traffic control"""

import numpy as np
import gymnasium as gym
from typing import Dict, Any
from collections import deque

from core.sar_framework import StateRepresentation, TrafficMetrics


class DensityFocusedState(StateRepresentation):
    """
    State representation focused on traffic density metrics.
    
    Features:
    - Traffic density (vehicles/km/lane)
    - Density gradient (change in density)
    - Critical density ratio
    - Speed-density correlation
    """
    
    def _setup(self):
        self.num_features = 5
        self.max_density = 150  # vehicles/km/lane
        self.critical_density = 30  # vehicles/km/lane for max flow
        self.density_history = deque(maxlen=10)
        
    def get_observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(
            low=np.zeros(self.num_features, dtype=np.float64),
            high=np.ones(self.num_features, dtype=np.float64),
            shape=(self.num_features,),
            dtype=np.float64
        )
    
    def build_state(self, metrics: TrafficMetrics) -> np.ndarray:
        # Calculate density from flow and speed
        if metrics.avg_speed_before > 0.1:  # Avoid division by zero
            density = metrics.flow_upstream / (metrics.avg_speed_before * 3.6)
        else:
            density = self.max_density  # Max density when stopped
        
        self.density_history.append(density)
        
        # Density gradient
        if len(self.density_history) >= 2:
            density_gradient = density - self.density_history[-2]
        else:
            density_gradient = 0
        
        # Critical density ratio
        critical_ratio = density / self.critical_density
        
        # Speed-density correlation
        if density > 0:
            speed_density_factor = metrics.avg_speed_before / (self.max_density / density)
        else:
            speed_density_factor = 1.0
        
        return np.array([
            density,
            density_gradient,
            critical_ratio,
            speed_density_factor,
            metrics.current_speed_limit
        ], dtype=np.float64)
    
    def preprocess_state(self, raw_state: np.ndarray) -> np.ndarray:
        return np.array([
            np.clip(raw_state[0], 0, self.max_density) / self.max_density,
            np.clip(raw_state[1], -50, 50) / 100 + 0.5,  # Gradient normalized to [0,1]
            np.clip(raw_state[2], 0, 3) / 3,  # Critical ratio capped at 3
            np.clip(raw_state[3], 0, 2) / 2,  # Speed-density factor
            np.clip(raw_state[4], 50, 130) / 130
        ], dtype=np.float64)


class MultiSegmentDensityState(StateRepresentation):
    """
    State representation using density from multiple road segments.
    
    Provides spatial awareness by tracking density across segments.
    """
    
    def _setup(self):
        self.num_segments = 3  # Monitor 3 upstream segments
        self.num_features = self.num_segments * 2 + 2  # Density + occupancy per segment + speed limit + avg flow
        self.max_density = 150
        
    def get_observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(
            low=np.zeros(self.num_features, dtype=np.float64),
            high=np.ones(self.num_features, dtype=np.float64),
            shape=(self.num_features,),
            dtype=np.float64
        )
    
    def build_state(self, metrics: TrafficMetrics) -> np.ndarray:
        # Simulate getting density from multiple segments
        # In real implementation, this would query SUMO for each segment
        base_density = metrics.flow_upstream / (max(metrics.avg_speed_before, 0.1) * 3.6)
        
        # Create artificial upstream densities (would be real SUMO data)
        densities = [
            base_density,
            base_density * 0.8,  # Upstream segment 1
            base_density * 0.6   # Upstream segment 2
        ]
        
        # Occupancies for each segment
        occupancies = [
            metrics.occupancy_upstream,
            metrics.occupancy_upstream * 0.8,
            metrics.occupancy_upstream * 0.6
        ]
        
        state = []
        for d, o in zip(densities, occupancies):
            state.extend([d, o])
        
        state.extend([
            metrics.current_speed_limit,
            metrics.flow_downstream
        ])
        
        return np.array(state, dtype=np.float64)
    
    def preprocess_state(self, raw_state: np.ndarray) -> np.ndarray:
        normalized = []
        
        # Normalize densities and occupancies
        for i in range(0, self.num_segments * 2, 2):
            normalized.append(np.clip(raw_state[i], 0, self.max_density) / self.max_density)
            normalized.append(np.clip(raw_state[i+1], 0, 100) / 100)
        
        # Normalize speed limit and flow
        normalized.append(np.clip(raw_state[-2], 50, 130) / 130)
        normalized.append(np.clip(raw_state[-1], 0, self.config.get('max_flow', 10000)) / self.config.get('max_flow', 10000))
        
        return np.array(normalized, dtype=np.float64)