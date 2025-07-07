#!/usr/bin/env python3
"""Quick fix for basic_*.py files"""

import os

files_to_fix = {
    'sar_components/states/basic_states.py': '''# sar_components/states/basic_states.py
"""
Basic state representations (FullMetricsState and MinimalState) are defined
in core.sar_framework. Please import them directly from there or use the
factory function create_state_representation().
"""

# This file is intentionally empty to avoid circular imports
''',
    
    'sar_components/actions/speed_actions.py': '''# sar_components/actions/speed_actions.py
"""
Basic action strategies (AbsoluteSpeedAction and RelativeSpeedAction) are defined
in core.sar_framework. Please import them directly from there or use the
factory function create_action_strategy().
"""

# This file is intentionally empty to avoid circular imports
''',
    
    'sar_components/rewards/basic_rewards.py': '''# sar_components/rewards/basic_rewards.py
"""
Basic reward functions (MobilityReward, SafetyReward, and BalancedReward) are defined
in core.sar_framework. Please import them directly from there or use the
factory function create_reward_function().
"""

# This file is intentionally empty to avoid circular imports
'''
}

for filepath, content in files_to_fix.items():
    with open(filepath, 'w') as f:
        f.write(content)
    print(f"✓ Fixed {filepath}")

print("\nNow check core/sar_framework.py and remove any MARVEL imports from the top!")