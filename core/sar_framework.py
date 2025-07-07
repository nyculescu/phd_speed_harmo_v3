# sar_framework.py
"""
Modular State-Action-Reward (SAR) Framework for DRL-VSL

This module provides abstract base classes and concrete implementations
for different state representations, action strategies, and reward functions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import gymnasium as gym
from collections import deque
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Data class to hold traffic metrics
@dataclass
class TrafficMetrics:
    """Container for all traffic measurements"""
    avg_speed_before: float = 0.0
    flow_upstream: float = 0.0
    flow_downstream: float = 0.0
    flow_smoothed: float = 0.0
    queue_length_upstream: float = 0.0
    occupancy_upstream: float = 0.0
    occupancy_smoothed: float = 0.0
    current_speed_limit: float = 130.0
    simulation_step: int = 0
    simulation_time: float = 0.0
    collisions_count: int = 0
    
    # Historical data
    flow_downstream_history: deque = None
    occupancy_downstream_history: deque = None
    speed_history: deque = None
    
    def __post_init__(self):
        if self.flow_downstream_history is None:
            self.flow_downstream_history = deque(maxlen=5)
        if self.occupancy_downstream_history is None:
            self.occupancy_downstream_history = deque(maxlen=5)
        if self.speed_history is None:
            self.speed_history = deque(maxlen=15)


# ============================================================================
# STATE REPRESENTATIONS
# ============================================================================

class StateRepresentation(ABC):
    """Abstract base class for state representations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """Initialize the state representation"""
        pass
    
    @abstractmethod
    def get_observation_space(self) -> gym.spaces.Space:
        """Return the observation space for this representation"""
        pass
    
    @abstractmethod
    def build_state(self, metrics: TrafficMetrics) -> np.ndarray:
        """Build the raw state vector from traffic metrics"""
        pass
    
    @abstractmethod
    def preprocess_state(self, raw_state: np.ndarray) -> np.ndarray:
        """Normalize/preprocess the raw state for the neural network"""
        pass
    
    def get_observation(self, metrics: TrafficMetrics) -> np.ndarray:
        """Get the preprocessed observation"""
        raw_state = self.build_state(metrics)
        return self.preprocess_state(raw_state)


class FullMetricsState(StateRepresentation):
    """Full 9-feature state representation"""
    
    def _setup(self):
        self.num_features = 9
        self.max_speed_mps = 130 / 3.6
        self.max_flow = self.config.get('max_flow', 10000.0)
        self.max_occupancy = self.config.get('max_occupancy', 100.0)
        self.max_queue_length = self.config.get('max_queue_length', 575.0 * 3 / 7)
        self.speed_trend_clip = 1.0
        self.time_since_last_action = 0
    
    def get_observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(
            low=np.zeros(self.num_features, dtype=np.float64),
            high=np.ones(self.num_features, dtype=np.float64),
            shape=(self.num_features,),
            dtype=np.float64
        )
    
    def build_state(self, metrics: TrafficMetrics) -> np.ndarray:
        return np.array([
            metrics.avg_speed_before,
            metrics.flow_upstream,
            metrics.flow_smoothed,
            metrics.queue_length_upstream,
            self._calculate_speed_trend(metrics),
            metrics.occupancy_smoothed / 100.0,
            metrics.current_speed_limit,
            self._calculate_speed_stability(metrics),
            self.time_since_last_action
        ], dtype=np.float64)
    
    def preprocess_state(self, raw_state: np.ndarray) -> np.ndarray:
        if len(raw_state) != self.num_features:
            logger.error(f"Mismatched raw_state length: {len(raw_state)} vs {self.num_features}")
            return np.zeros(self.num_features, dtype=np.float64)
        
        avg_speed = np.clip(raw_state[0], 0, self.max_speed_mps) / self.max_speed_mps
        flow_upstream = np.clip(raw_state[1], 0, self.max_flow) / self.max_flow
        flow_smoothed = np.clip(raw_state[2], 0, self.max_flow) / self.max_flow
        queue_length = np.clip(raw_state[3], 0, self.max_queue_length) / self.max_queue_length
        
        speed_trend = np.clip(raw_state[4], -self.speed_trend_clip, self.speed_trend_clip)
        speed_trend_norm = (speed_trend + self.speed_trend_clip) / (2 * self.speed_trend_clip)
        
        occupancy = np.clip(raw_state[5], 0, 1)
        speed_limit = np.clip(raw_state[6], 50, 130) / 130.0
        speed_stability_norm = raw_state[7]
        time_since_change_norm = min(raw_state[8] / 10.0, 1.0)
        
        return np.array([
            avg_speed, flow_upstream, flow_smoothed, queue_length,
            speed_trend_norm, occupancy, speed_limit, speed_stability_norm,
            time_since_change_norm
        ], dtype=np.float64)
    
    def _calculate_speed_trend(self, metrics: TrafficMetrics) -> float:
        if len(metrics.speed_history) < 3:
            return 0.0
        
        speeds = list(metrics.speed_history)
        n = len(speeds)
        x = np.arange(n)
        
        x_mean = np.mean(x)
        y_mean = np.mean(speeds)
        
        numerator = np.sum((x - x_mean) * (speeds - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return float(slope)
    
    def _calculate_speed_stability(self, metrics: TrafficMetrics) -> float:
        if len(metrics.speed_history) > 5:
            std_dev = np.std(list(metrics.speed_history))
            return float(1.0 / (1.0 + std_dev))
        return 0.5


class MinimalState(StateRepresentation):
    """Minimal 3-feature state representation"""
    
    def _setup(self):
        self.num_features = 3
        self.max_flow = self.config.get('max_flow', 10000.0)
    
    def get_observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(
            low=np.zeros(self.num_features, dtype=np.float64),
            high=np.ones(self.num_features, dtype=np.float64),
            shape=(self.num_features,),
            dtype=np.float64
        )
    
    def build_state(self, metrics: TrafficMetrics) -> np.ndarray:
        return np.array([
            metrics.flow_smoothed,
            metrics.occupancy_smoothed,
            metrics.current_speed_limit
        ], dtype=np.float64)
    
    def preprocess_state(self, raw_state: np.ndarray) -> np.ndarray:
        return np.array([
            np.clip(raw_state[0], 0, self.max_flow) / self.max_flow,
            np.clip(raw_state[1], 0, 100) / 100.0,
            np.clip(raw_state[2], 50, 130) / 130.0
        ], dtype=np.float64)


# ============================================================================
# ACTION STRATEGIES
# ============================================================================

class ActionStrategy(ABC):
    """Abstract base class for action strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.invalid_action_penalty = 0.0
        self._setup()
    
    @abstractmethod
    def _setup(self):
        """Initialize the action strategy"""
        pass
    
    @abstractmethod
    def get_action_space(self) -> gym.spaces.Space:
        """Return the action space for this strategy"""
        pass
    
    @abstractmethod
    def apply_action(self, action: int, current_speed_limit: float) -> Tuple[float, float]:
        """
        Apply the action and return (new_speed_limit, penalty)
        """
        pass


class AbsoluteSpeedAction(ActionStrategy):
    """Actions directly set the speed limit"""
    
    def _setup(self):
        self.speed_actions = {
            0: 60, 1: 70, 2: 80, 3: 90,
            4: 100, 5: 110, 6: 120, 7: 130
        }
    
    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(len(self.speed_actions))
    
    def apply_action(self, action: int, current_speed_limit: float) -> Tuple[float, float]:
        proposed_speed_limit = self.speed_actions.get(action, current_speed_limit)
        
        penalty = 0.0
        speed_change = abs(proposed_speed_limit - current_speed_limit)
        if speed_change > 20:  # Penalize jumps larger than 20 km/h
            penalty = -0.2 * (speed_change / 10)
        
        return proposed_speed_limit, penalty


class RelativeSpeedAction(ActionStrategy):
    """Actions change speed limit relative to current"""
    
    def _setup(self):
        self.speed_changes = [-20, -10, -5, 0, 5, 10, 20]
        self.recent_changes = deque(maxlen=5)
    
    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(len(self.speed_changes))
    
    def apply_action(self, action: int, current_speed_limit: float) -> Tuple[float, float]:
        if action >= len(self.speed_changes):
            return current_speed_limit, -1.0
        
        speed_change = self.speed_changes[action]
        proposed_speed_limit = current_speed_limit + speed_change
        penalty = 0.0
        
        # Safety constraint: limit consecutive changes
        if len(self.recent_changes) >= 3:
            if all(abs(change) >= 5 for change in list(self.recent_changes)[-3:]):
                proposed_speed_limit = current_speed_limit
                penalty = -2.0
        
        # Apply bounds
        proposed_speed_limit = max(50, min(130, proposed_speed_limit))
        
        # Track changes
        self.recent_changes.append(proposed_speed_limit - current_speed_limit)
        
        return proposed_speed_limit, penalty


# ============================================================================
# REWARD FUNCTIONS
# ============================================================================

class RewardFunction(ABC):
    """Abstract base class for reward functions"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_flow = config.get('max_flow', 10000.0)
        self.max_occupancy = config.get('max_occupancy', 100.0)
        self.max_queue_length = config.get('max_queue_length', 575.0 * 3 / 7)
    
    @abstractmethod
    def calculate(self, metrics: TrafficMetrics, 
                  action_penalty: float = 0.0,
                  collision_penalty: float = 0.0) -> float:
        """Calculate the reward based on current metrics"""
        pass


class MobilityReward(RewardFunction):
    """Mobility-focused reward based on maintaining optimal density"""
    
    def calculate(self, metrics: TrafficMetrics, 
                  action_penalty: float = 0.0,
                  collision_penalty: float = 0.0) -> float:
        
        # Parameters based on traffic flow theory
        CRITICAL_OCCUPANCY = 28.0
        TARGET_OCCUPANCY_UPPER_BOUND = 32.0
        CONGESTION_THRESHOLD = 50.0
        REWARD_SCALE = 0.02
        BONUS_REWARD = 0.6
        CONGESTION_PENALTY = -0.6
        
        current_occupancy = metrics.occupancy_smoothed
        base_reward = 0.0
        
        # Triangular reward function peaking at critical occupancy
        if current_occupancy < CRITICAL_OCCUPANCY:
            base_reward = REWARD_SCALE * current_occupancy
        else:
            denominator = (CONGESTION_THRESHOLD - CRITICAL_OCCUPANCY)
            if denominator > 0:
                max_reward = REWARD_SCALE * CRITICAL_OCCUPANCY
                base_reward = max_reward - (max_reward * (current_occupancy - CRITICAL_OCCUPANCY) / denominator)
            else:
                base_reward = -REWARD_SCALE * current_occupancy
        
        base_reward = max(0.0, base_reward)
        
        # Bonus for being in the sweet spot
        if CRITICAL_OCCUPANCY <= current_occupancy <= TARGET_OCCUPANCY_UPPER_BOUND:
            base_reward += BONUS_REWARD
        
        # Penalty for severe congestion
        if current_occupancy > CONGESTION_THRESHOLD:
            base_reward += CONGESTION_PENALTY
        
        return float(base_reward + collision_penalty + action_penalty)


class SafetyReward(RewardFunction):
    """Safety-focused reward emphasizing speed harmonization"""
    
    def calculate(self, metrics: TrafficMetrics,
                  action_penalty: float = 0.0,
                  collision_penalty: float = 0.0) -> float:
        
        # Speed harmonization (variance reduction)
        R_safety_variance = self._calculate_speed_smoothness(metrics) * 0.5
        
        # Maintain speed in optimal safety band
        target_speed_lower = 80 / 3.6
        target_speed_upper = 100 / 3.6
        
        if target_speed_lower <= metrics.avg_speed_before <= target_speed_upper:
            R_safety_speed_band = 0.3
        else:
            distance_from_band = min(
                abs(metrics.avg_speed_before - target_speed_lower),
                abs(metrics.avg_speed_before - target_speed_upper)
            )
            R_safety_speed_band = -min((distance_from_band / (130/3.6))**2, 1.0) * 0.3
        
        # Small flow incentive
        R_flow = min(metrics.flow_smoothed / self.max_flow, 1.0) * 0.1
        
        # Enhanced collision penalty
        enhanced_collision_penalty = collision_penalty * 2
        
        return float(R_safety_variance + R_safety_speed_band + R_flow + 
                     enhanced_collision_penalty + action_penalty)
    
    def _calculate_speed_smoothness(self, metrics: TrafficMetrics) -> float:
        if len(metrics.speed_history) < 2:
            return 0.0
        
        speed_variance = np.var(list(metrics.speed_history))
        max_variance = 400.0
        smoothness = max(0.0, 1.0 - (speed_variance / max_variance))
        return smoothness


class BalancedReward(RewardFunction):
    """Balanced reward combining multiple objectives"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.previous_speed_limit = 130.0
    
    def calculate(self, metrics: TrafficMetrics,
                  action_penalty: float = 0.0,
                  collision_penalty: float = 0.0) -> float:
        
        # Flow component
        R_flow = min(metrics.flow_smoothed / self.max_flow, 1.0) * 0.3
        
        # Safety component
        if len(metrics.speed_history) > 5:
            speed_variance = np.var(list(metrics.speed_history))
            R_safety = max(0.0, np.exp(-speed_variance / 200.0)) * 0.4
        else:
            R_safety = 0.2
        
        # Control smoothness
        speed_change_magnitude = abs(metrics.current_speed_limit - self.previous_speed_limit)
        R_smoothness = max(0.0, 1.0 - (speed_change_magnitude / 15.0)**1.5) * 0.15
        
        # Efficiency component
        target_speed = 100.0 / 3.6
        if metrics.avg_speed_before > 0:
            speed_efficiency = 1.0 - abs(metrics.avg_speed_before - target_speed) / target_speed
            flow_speed_synergy = min(metrics.flow_smoothed / self.max_flow, 1.0) * max(0, speed_efficiency)
            R_efficiency = (max(0.0, speed_efficiency) * 0.1) + (flow_speed_synergy * 0.05)
        else:
            R_efficiency = 0.0
        
        # Queue penalty
        if metrics.queue_length_upstream > 0:
            queue_ratio = metrics.queue_length_upstream / self.max_queue_length
            queue_penalty = min(queue_ratio**2 * 0.2, 0.2)
        else:
            queue_penalty = 0.0
        
        # Progress bonus
        progress_bonus = min(metrics.simulation_step / 5000, 1.0) * 0.05
        
        self.previous_speed_limit = metrics.current_speed_limit
        
        return float(R_flow + R_safety + R_smoothness + R_efficiency + 
                     progress_bonus - queue_penalty + action_penalty + collision_penalty)


# ============================================================================
# FACTORY FUNCTIONS with LAZY IMPORTS
# ============================================================================

def create_state_representation(name: str, config: Dict[str, Any]) -> StateRepresentation:
    """Factory function to create state representations"""
    representations = {
        'full_metrics': FullMetricsState,
        'minimal': MinimalState,
    }
    
    # Lazy import MARVEL to avoid circular dependency
    if name == 'marvel':
        from sar_components.states.marvel_states import MARVELState
        representations['marvel'] = MARVELState
    
    if name not in representations:
        raise ValueError(f"Unknown state representation: {name}")
    
    return representations[name](config)


def create_action_strategy(name: str, config: Dict[str, Any]) -> ActionStrategy:
    """Factory function to create action strategies"""
    strategies = {
        'absolute_speed': AbsoluteSpeedAction,
        'relative_speed': RelativeSpeedAction,
    }
    
    # Lazy import MARVEL to avoid circular dependency
    if name == 'marvel_speed':
        from sar_components.actions.marvel_actions import MARVELSpeedAction
        strategies['marvel_speed'] = MARVELSpeedAction
    
    if name not in strategies:
        raise ValueError(f"Unknown action strategy: {name}")
    
    return strategies[name](config)


def create_reward_function(name: str, config: Dict[str, Any]) -> RewardFunction:
    """Factory function to create reward functions"""
    functions = {
        'mobility': MobilityReward,
        'safety': SafetyReward,
        'balanced': BalancedReward,
    }
    
    # Lazy import MARVEL to avoid circular dependency
    if name == 'marvel':
        from sar_components.rewards.marvel_rewards import MARVELReward
        functions['marvel'] = MARVELReward
    
    if name not in functions:
        raise ValueError(f"Unknown reward function: {name}")
    
    return functions[name](config)