# sar_components/__init__.py
"""
SAR Components Library

Collection of State representations, Action strategies, and Reward functions
for the DRL-VSL framework.
"""

# Import all states
from .states import (
    # Basic states
    FullMetricsState,
    MinimalState,
    # Density-focused states
    DensityFocusedState,
    MultiSegmentDensityState,
    # Queue-focused states
    QueueLengthState,
    QueueDynamicsState,
)

# Import all actions
from .actions import (
    # Speed actions
    AbsoluteSpeedAction,
    RelativeSpeedAction,
    # Adaptive actions
    TrafficAdaptiveAction,
    OccupancyBasedAction,
    # Smooth actions
    GradualChangeAction,
    MomentumBasedAction,
)

# Import all rewards
from .rewards import (
    # Basic rewards
    MobilityReward,
    SafetyReward,
    BalancedReward,
    # Emission rewards
    EmissionReward,
    FuelEfficiencyReward,
    # Comfort rewards
    PassengerComfortReward,
    SmoothFlowReward,
)

__all__ = [
    # States
    "FullMetricsState", "MinimalState",
    "DensityFocusedState", "MultiSegmentDensityState",
    "QueueLengthState", "QueueDynamicsState",
    
    # Actions
    "AbsoluteSpeedAction", "RelativeSpeedAction",
    "TrafficAdaptiveAction", "OccupancyBasedAction",
    "GradualChangeAction", "MomentumBasedAction",
    
    # Rewards
    "MobilityReward", "SafetyReward", "BalancedReward",
    "EmissionReward", "FuelEfficiencyReward",
    "PassengerComfortReward", "SmoothFlowReward",
]