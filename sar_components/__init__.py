# sar_components/__init__.py
"""
SAR Components Package

This package contains additional state representations, action strategies,
and reward functions for the DRL-VSL framework.

Note: Basic components (FullMetricsState, MinimalState, AbsoluteSpeedAction,
RelativeSpeedAction, MobilityReward, SafetyReward, BalancedReward) are 
defined in core.sar_framework to avoid circular imports.
"""

# Import only the additional components defined in this package
# MARVEL components
from .states.marvel_states import MARVELState
from .actions.marvel_actions import MARVELSpeedAction
from .rewards.marvel_rewards import MARVELReward

# Additional reward functions
from .rewards.emission_rewards import EmissionReward, FuelEfficiencyReward
from .rewards.comfort_rewards import PassengerComfortReward, SmoothFlowReward

# DQN-VSL components
from .states.kang15_states import KANG15State
from .actions.kang15_actions import KANG15Action
from .rewards.kang15_rewards import KANG15Reward

__all__ = [
    # MARVEL components
    "MARVELState",
    "MARVELSpeedAction", 
    "MARVELReward",
    
    # Additional rewards
    "EmissionReward",
    "FuelEfficiencyReward",
    "PassengerComfortReward",
    "SmoothFlowReward",

    # DQN-VSL components
    "KANG15State",
    "KANG15Action",
    "KANG15Reward"
]
