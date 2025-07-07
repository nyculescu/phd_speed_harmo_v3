# core/__init__.py
"""
Core DRL-VSL Framework

This package contains the core components for the modular State-Action-Reward framework
and the refactored traffic environment.
"""

# Version
__version__ = "1.1.0"

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
    create_sumocfg
)

# Import integration/compatibility layer
from .drl_vsl_integration import (
    DEFAULT_SAR_CONFIG,
    load_sar_config_from_file,
    get_model_config_path,
    merge_sar_configs,
    PRESET_SAR_CONFIGS,
    get_preset_sar_config,
    create_traffic_env_from_config,
    create_train_env_helper,
    create_eval_env_helper,
)

# SUMO configuration utilities
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
    # "MARVELState", NOTE: MARVELState is available through factory function
    
    # Action strategies
    "AbsoluteSpeedAction",
    "RelativeSpeedAction",
    # "MARVELSpeedAction", NOTE: MARVELSpeedAction is available through factory function
    
    # Reward functions
    "MobilityReward",
    "SafetyReward",
    "BalancedReward",
    # "MARVELReward", NOTE: MARVELReward is available through factory function
    
    # Factory functions
    "create_state_representation",
    "create_action_strategy",
    "create_reward_function",
    
    # Environments
    "TrafficEnv",
    "TrafficDataLogger",
    "create_sumocfg",
    
    # Integration utilities
    "DEFAULT_SAR_CONFIG",
    "load_sar_config_from_file",
    "get_model_config_path",
    "merge_sar_configs",
    "PRESET_SAR_CONFIGS",
    "get_preset_sar_config",
    "create_traffic_env_from_config",
    "create_train_env_helper",
    "create_eval_env_helper",
    
    # SUMO configuration
    "SumoConfig",
    "load_sumo_config",
    "get_default_sumo_config",
    "get_preset_config",
    "PRESETS",
]