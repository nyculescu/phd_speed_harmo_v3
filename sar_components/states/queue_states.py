# sar_components/states/queue_states.py
"""Queue-focused state representations for congestion management"""

import numpy as np
import gymnasium as gym
from collections import deque

from core.sar_framework import StateRepresentation, TrafficMetrics


class QueueLengthState(StateRepresentation):
    """
    Simple queue-focused state representation.
    
    Features:
    - Current queue length
    - Queue growth rate
    - Queue-to-capacity ratio
    - Time since queue formed
    """
    
    def _setup(self):
        self.num_features = 4
        self.max_queue_length = self.config.get('max_queue_length', 575.0 * 3 / 7)
        self.queue_history = deque(maxlen=5)
        self.queue_start_time = None
        self.queue_threshold = 50  # meters
        
    def get_observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(
            low=np.zeros(self.num_features, dtype=np.float64),
            high=np.ones(self.num_features, dtype=np.float64),
            shape=(self.num_features,),
            dtype=np.float64
        )
    
    def build_state(self, metrics: TrafficMetrics) -> np.ndarray:
        current_queue = metrics.queue_length_upstream
        self.queue_history.append(current_queue)
        
        # Queue growth rate
        if len(self.queue_history) >= 2:
            queue_growth = current_queue - self.queue_history[-2]
        else:
            queue_growth = 0
        
        # Track queue duration
        if current_queue > self.queue_threshold:
            if self.queue_start_time is None:
                self.queue_start_time = metrics.simulation_time
            queue_duration = metrics.simulation_time - self.queue_start_time
        else:
            self.queue_start_time = None
            queue_duration = 0
        
        # Queue-to-capacity ratio
        queue_ratio = current_queue / self.max_queue_length
        
        return np.array([
            current_queue,
            queue_growth,
            queue_ratio,
            queue_duration
        ], dtype=np.float64)
    
    def preprocess_state(self, raw_state: np.ndarray) -> np.ndarray:
        return np.array([
            np.clip(raw_state[0], 0, self.max_queue_length) / self.max_queue_length,
            np.clip(raw_state[1], -100, 100) / 200 + 0.5,  # Growth rate
            np.clip(raw_state[2], 0, 2),  # Ratio can exceed 1
            np.clip(raw_state[3], 0, 3600) / 3600  # Max 1 hour
        ], dtype=np.float64)


class QueueDynamicsState(StateRepresentation):
    """
    Advanced queue state with discharge rate and shockwave analysis.
    
    Features:
    - Queue length and growth
    - Discharge rate at queue head
    - Shockwave speed estimate
    - Queue density
    - Spillback risk
    """
    
    def _setup(self):
        self.num_features = 6
        self.max_queue_length = self.config.get('max_queue_length', 575.0 * 3 / 7)
        self.queue_history = deque(maxlen=10)
        self.discharge_history = deque(maxlen=5)
        self.segment_length = 500  # meters
        
    def get_observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(
            low=np.zeros(self.num_features, dtype=np.float64),
            high=np.ones(self.num_features, dtype=np.float64),
            shape=(self.num_features,),
            dtype=np.float64
        )
    
    def build_state(self, metrics: TrafficMetrics) -> np.ndarray:
        current_queue = metrics.queue_length_upstream
        self.queue_history.append(current_queue)
        
        # Queue growth rate
        if len(self.queue_history) >= 2:
            queue_growth = current_queue - self.queue_history[-2]
        else:
            queue_growth = 0
        
        # Estimate discharge rate (vehicles leaving queue)
        if metrics.avg_speed_before > 5:  # Queue is discharging
            discharge_rate = metrics.flow_downstream
        else:
            discharge_rate = 0
        self.discharge_history.append(discharge_rate)
        
        # Average discharge rate
        avg_discharge = np.mean(self.discharge_history) if self.discharge_history else 0
        
        # Shockwave speed estimate (simplified)
        if queue_growth != 0 and metrics.flow_upstream > 0:
            shockwave_speed = abs(queue_growth) / 60  # m/s
        else:
            shockwave_speed = 0
        
        # Queue density
        if current_queue > 0:
            # Assume average vehicle length of 7m
            queue_density = (current_queue / 7) / (current_queue / 1000)  # vehicles/km
        else:
            queue_density = 0
        
        # Spillback risk (queue approaching segment boundary)
        spillback_risk = current_queue / self.segment_length
        
        return np.array([
            current_queue,
            queue_growth,
            avg_discharge,
            shockwave_speed,
            queue_density,
            spillback_risk
        ], dtype=np.float64)
    
    def preprocess_state(self, raw_state: np.ndarray) -> np.ndarray:
        return np.array([
            np.clip(raw_state[0], 0, self.max_queue_length) / self.max_queue_length,
            np.clip(raw_state[1], -100, 100) / 200 + 0.5,
            np.clip(raw_state[2], 0, self.config.get('max_flow', 10000)) / self.config.get('max_flow', 10000),
            np.clip(raw_state[3], 0, 20) / 20,  # Max shockwave speed 20 m/s
            np.clip(raw_state[4], 0, 200) / 200,  # Max density
            np.clip(raw_state[5], 0, 2)  # Spillback risk can exceed 1
        ], dtype=np.float64)