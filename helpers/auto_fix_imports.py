#!/usr/bin/env python3
"""
auto_fix_imports.py - Automatically fix all import issues
"""

from pathlib import Path

def fix_sar_framework():
    """Fix the lazy imports in sar_framework.py"""
    
    sar_file = Path("core/sar_framework.py")
    
    if not sar_file.exists():
        print(f"Error: {sar_file} not found!")
        return False
    
    # Read the file
    with open(sar_file, 'r') as f:
        lines = f.readlines()
    
    # Remove any MARVEL imports from the top of the file
    new_lines = []
    in_imports_section = True
    removed_imports = []
    
    for line in lines:
        # Check if we're still in the imports section (before class definitions)
        if line.strip().startswith('class ') or line.strip().startswith('def '):
            in_imports_section = False
        
        # Remove MARVEL imports only in the imports section
        if in_imports_section and ('marvel' in line.lower() and 
            ('from sar_components' in line or 'import' in line)):
            removed_imports.append(line.strip())
            continue
        
        new_lines.append(line)
    
    # Now fix the factory functions
    # We need to add the actual import statements inside the if blocks
    
    # Process line by line to fix the factory functions
    fixed_lines = []
    i = 0
    while i < len(new_lines):
        line = new_lines[i]
        
        # Fix create_state_representation
        if "representations['marvel'] = MARVELState" in line:
            # Add the import before this line
            indent = len(line) - len(line.lstrip())
            fixed_lines.append(' ' * indent + 'from sar_components.states.marvel_states import MARVELState\n')
            fixed_lines.append(line)
        
        # Fix create_action_strategy  
        elif "strategies['marvel_speed'] = MARVELSpeedAction" in line:
            # Add the import before this line
            indent = len(line) - len(line.lstrip())
            fixed_lines.append(' ' * indent + 'from sar_components.actions.marvel_actions import MARVELSpeedAction\n')
            fixed_lines.append(line)
        
        # Fix create_reward_function
        elif "functions['marvel'] = MARVELReward" in line:
            # Add the import before this line
            indent = len(line) - len(line.lstrip())
            fixed_lines.append(' ' * indent + 'from sar_components.rewards.marvel_rewards import MARVELReward\n')
            fixed_lines.append(line)
        
        else:
            fixed_lines.append(line)
        
        i += 1
    
    # Write the fixed content back
    with open(sar_file, 'w') as f:
        f.writelines(fixed_lines)
    
    print(f"✓ Fixed {sar_file}")
    if removed_imports:
        print("  - Removed MARVEL imports from top of file:")
        for imp in removed_imports:
            print(f"    {imp}")
    print("  - Added proper lazy imports in factory functions")
    
    return True

if __name__ == "__main__":
    success = fix_sar_framework()
    if success:
        print("\nNow run: python check_integration.py")
    else:
        print("\nFix failed!")