#!/usr/bin/env python3
"""
check_integration.py - Verify MARVEL integration is complete
Version 2: Smarter about checking imports (ignores comments)
"""

import sys
import os
import re
from pathlib import Path

def check_file_for_import(filepath, import_pattern, should_find=True):
    """Check if a file contains an actual import (not in comments)."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        found = False
        for line in lines:
            # Skip comment lines
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            # Skip lines that are in docstrings (rough check)
            if '"""' in line or "'''" in line:
                continue
            # Check for the import
            if import_pattern in line and not line.strip().startswith('#'):
                found = True
                break
        
        return found == should_find
    except Exception as e:
        print(f"  ✗ Error reading {filepath}: {e}")
        return False

def check_file_contains(filepath, search_string, should_contain=True):
    """Check if a file contains (or doesn't contain) a string."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        found = search_string in content
        return found == should_contain
    except Exception as e:
        print(f"  ✗ Error reading {filepath}: {e}")
        return False

def check_integration():
    """Run all integration checks."""
    print("=" * 60)
    print("MARVEL Integration Checker v2")
    print("=" * 60)
    
    all_good = True
    
    # 1. Check core/sar_framework.py has lazy imports
    print("\n1. Checking core/sar_framework.py...")
    sar_file = Path("core/sar_framework.py")
    
    if sar_file.exists():
        # Should NOT have MARVEL imports at top (not in comments)
        marvel_imports = [
            "from sar_components.states.marvel_states import",
            "from sar_components.actions.marvel_actions import",
            "from sar_components.rewards.marvel_rewards import"
        ]
        
        has_top_imports = False
        for imp in marvel_imports:
            if not check_file_for_import(sar_file, imp, False):
                pass  # Good, no import
            else:
                has_top_imports = True
                break
        
        if not has_top_imports:
            print("  ✓ No MARVEL imports at top of file")
        else:
            print("  ✗ Found MARVEL imports at top - should use lazy imports in factory functions")
            all_good = False
            
        # Should have lazy import in factory
        if check_file_contains(sar_file, "if name == 'marvel':", True):
            print("  ✓ Has lazy import for MARVEL in factory function")
        else:
            print("  ✗ Missing lazy import for MARVEL in factory function")
            all_good = False
    else:
        print("  ✗ File not found!")
        all_good = False
    
    # 2. Check sar_components files are updated
    print("\n2. Checking sar_components updates...")
    basic_files = [
        "sar_components/states/basic_states.py",
        "sar_components/actions/speed_actions.py",
        "sar_components/rewards/basic_rewards.py"
    ]
    
    for bf in basic_files:
        if Path(bf).exists():
            if check_file_for_import(bf, "from core.sar_framework import", False):
                print(f"  ✓ {bf} - no circular imports")
            else:
                print(f"  ✗ {bf} - still has active imports from core.sar_framework")
                all_good = False
        else:
            print(f"  ⚠ {bf} - not found (might be okay if deleted)")
    
    # 3. Check MARVEL files use km/h
    print("\n3. Checking MARVEL unit conversions...")
    marvel_action = Path("sar_components/actions/marvel_actions.py")
    
    if marvel_action.exists():
        if check_file_contains(marvel_action, "48.3", True):  # 30 mph in km/h
            print("  ✓ MARVEL actions use km/h")
        else:
            print("  ✗ MARVEL actions still use mph")
            all_good = False
    else:
        print("  ✗ marvel_actions.py not found!")
        all_good = False
    
    # 4. Check drl_vsl_train.py has determine_configurations
    print("\n4. Checking drl_vsl_train.py updates...")
    train_file = Path("training/drl_vsl_train.py")
    
    if train_file.exists():
        if check_file_contains(train_file, "def determine_configurations", True):
            print("  ✓ Has determine_configurations function")
        else:
            print("  ✗ Missing determine_configurations function")
            all_good = False
            
        if check_file_contains(train_file, "--combo-name", False):
            print("  ✓ No --combo-name argument (good!)")
        else:
            print("  ✗ Still has --combo-name argument")
            all_good = False
    else:
        print("  ✗ training/drl_vsl_train.py not found!")
        all_good = False
    
    # 5. Check YAML configuration
    print("\n5. Checking YAML configuration...")
    yaml_file = Path("training/config/drl_vsl_train_config.yaml")
    
    if yaml_file.exists():
        try:
            import yaml
            with open(yaml_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check structure
            if 'execution' in config:
                print("  ✓ Has execution section at top level")
            else:
                print("  ✗ Missing execution section")
                all_good = False
                
            if 'execution' in config.get('sar_config', {}):
                print("  ✗ execution is inside sar_config (should be at top level)")
                all_good = False
            else:
                print("  ✓ execution is not inside sar_config")
                
            if 'custom_combinations' in config.get('sar_config', {}):
                print(f"  ✓ Has {len(config['sar_config']['custom_combinations'])} custom combinations")
            else:
                print("  ✗ No custom_combinations defined")
                all_good = False
                
        except Exception as e:
            print(f"  ✗ Error parsing YAML: {e}")
            all_good = False
    else:
        print("  ✗ YAML config not found!")
        all_good = False
    
    # 6. Test imports
    print("\n6. Testing Python imports...")
    try:
        from core.sar_framework import create_state_representation
        print("  ✓ Can import create_state_representation")
        
        # Try to create MARVEL
        marvel_state = create_state_representation('marvel', {'max_flow': 10000})
        print("  ✓ Can create MARVEL state")
        
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        all_good = False
    
    # 7. Additional check - test all factory functions
    print("\n7. Testing all factory functions...")
    try:
        from core.sar_framework import (
            create_state_representation,
            create_action_strategy,
            create_reward_function
        )
        
        # Test creating each MARVEL component
        test_config = {'max_flow': 10000, 'max_occupancy': 100, 'max_queue_length': 246.4}
        
        marvel_state = create_state_representation('marvel', test_config)
        print("  ✓ Created MARVEL state")
        
        marvel_action = create_action_strategy('marvel_speed', test_config)
        print("  ✓ Created MARVEL action")
        
        marvel_reward = create_reward_function('marvel', test_config)
        print("  ✓ Created MARVEL reward")
        
    except Exception as e:
        print(f"  ✗ Error creating MARVEL components: {e}")
        all_good = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_good:
        print("✅ ALL CHECKS PASSED! Integration is complete.")
        print("\nYou can now run:")
        print("  python training/drl_vsl_train.py --list-combos")
        print("  python training/drl_vsl_train.py")
    else:
        print("❌ SOME CHECKS FAILED. Please fix the issues above.")
        print("\nQuick fix:")
        print("  1. Remove MARVEL imports from top of core/sar_framework.py")
        print("  2. Run: python fix_basic_files.py")
    print("=" * 60)
    
    return all_good

if __name__ == "__main__":
    success = check_integration()
    sys.exit(0 if success else 1)