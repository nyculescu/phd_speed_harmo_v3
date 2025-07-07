#!/usr/bin/env python3
"""
auto_fix_imports.py - Automatically fix all import issues
"""

import os
import re
from pathlib import Path

def fix_sar_framework():
    """Remove MARVEL imports from top of core/sar_framework.py"""
    filepath = Path("core/sar_framework.py")
    
    if not filepath.exists():
        print(f"✗ {filepath} not found!")
        return False
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Lines to remove
    imports_to_remove = [
        "from sar_components.states.marvel_states import MARVELState",
        "from sar_components.actions.marvel_actions import MARVELSpeedAction",
        "from sar_components.rewards.marvel_rewards import MARVELReward",
    ]
    
    # Filter out the import lines
    new_lines = []
    removed_count = 0
    
    for line in lines:
        should_keep = True
        for imp in imports_to_remove:
            if imp in line:
                should_keep = False
                removed_count += 1
                break
        if should_keep:
            new_lines.append(line)
    
    # Write back
    with open(filepath, 'w') as f:
        f.writelines(new_lines)
    
    if removed_count > 0:
        print(f"✓ Removed {removed_count} MARVEL imports from {filepath}")
    else:
        print(f"✓ No MARVEL imports found in {filepath} (already clean)")
    
    return True

def fix_basic_files():
    """Fix all basic_*.py files to remove circular imports"""
    
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
        try:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✓ Fixed {filepath}")
        except Exception as e:
            print(f"✗ Error fixing {filepath}: {e}")
            return False
    
    return True

def main():
    """Run all fixes"""
    print("=" * 60)
    print("Auto-fixing MARVEL Integration Issues")
    print("=" * 60)
    
    print("\n1. Fixing core/sar_framework.py...")
    fix_sar_framework()
    
    print("\n2. Fixing basic_*.py files...")
    fix_basic_files()
    
    print("\n" + "=" * 60)
    print("✅ All fixes applied!")
    print("\nNow run the checker to verify:")
    print("  python helpers/check_integration.py")
    print("=" * 60)

if __name__ == "__main__":
    main()