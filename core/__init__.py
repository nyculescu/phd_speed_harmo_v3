# core/__init__.py
"""
Core DRL-VSL Framework

This package contains the core components for the modular State-Action-Reward framework
and the refactored traffic environment.
"""

# Version
__version__ = "0.1.0"

# Import base classes and interfaces from SAR framework
from .sar_framework import (
    # Base classes
    StateRepresentation,
    ActionStrategy,
    RewardFunction,
    TrafficMetrics,
    
    # Concrete implementations
    FullMetricsState,
    MinimalState,
    AbsoluteSpeedAction,
    RelativeSpeedAction,
    MobilityReward,
    SafetyReward,
    BalancedReward,
    
    # Factory functions
    create_state_representation,
    create_action_strategy,
    create_reward_function,
)

# Import refactored environment
from .drl_vsl import (
    TrafficEnv,
    TrafficDataLogger,
)

# Import integration/compatibility layer
from .drl_vsl_integration import (
    TrafficEnvCompat,
    create_train_env_compat,
    create_eval_env_compat,
    create_mobility_env,
    create_safety_env,
    create_custom_env,
    update_train_model_minimal,
    DEFAULT_SAR_CONFIG,
)

# SUMO configuration utilities
from .drl_vsl import create_sumocfg

# Expose main classes at package level

# SUMO configuration
try:
    from ..traffic_environment.sumo_config import (
        SumoConfig,
        load_sumo_config,
        get_default_sumo_config,
        get_preset_config,
        PRESETS
    )
except ImportError:
    # Sumo config module not available yet
    SumoConfig = None
    load_sumo_config = None
    get_default_sumo_config = None
    get_preset_config = None
    PRESETS = None

__all__ = [
    # Version
    "__version__",
    
    # Base classes
    "StateRepresentation",
    "ActionStrategy", 
    "RewardFunction",
    "TrafficMetrics",
    
    # State representations
    "FullMetricsState",
    "MinimalState",
    
    # Action strategies
    "AbsoluteSpeedAction",
    "RelativeSpeedAction",
    
    # Reward functions
    "MobilityReward",
    "SafetyReward",
    "BalancedReward",
    
    # Factory functions
    "create_state_representation",
    "create_action_strategy",
    "create_reward_function",
    
    # Environments
    "TrafficEnv",
    "TrafficEnvCompat",
    "TrafficDataLogger",
    
    # Helper functions
    "create_train_env_compat",
    "create_eval_env_compat",
    "create_mobility_env",
    "create_safety_env",
    "create_custom_env",
    "update_train_model_minimal",
    "create_sumocfg",
    
    # Configuration
    "DEFAULT_SAR_CONFIG",
]