# Migration Notes: DRL-VSL Monolithic to Modular SAR Framework

## Overview

This document provides detailed migration notes for transitioning from the monolithic `drl_vsl.py` implementation to the modular SAR (State-Action-Reward) framework. It covers specific code changes, common pitfalls, and validation strategies.

## Version Mapping

| Original File | New Files | Status |
|--------------|-----------|---------|
| drl_vsl.py | drl_vsl_refactored.py + sar_framework.py | Core functionality migrated |
| - | drl_vsl_integration.py | Backward compatibility layer |
| drl_vsl_hp-tune.py | (needs update) | Pending migration |
| flow_gen.py | (unchanged) | No changes needed |

## Critical Dependencies

### SUMO Configuration File Creation

**Issue**: The refactored code requires explicit `.sumocfg` file creation.

**Original Code** (in drl_vsl.py):
```python
# Was handled implicitly in train_env_constructor
create_sumocfg(f"{model_name}_{idx}")
```

**New Code** (must add to refactored version):
```python
# In TrafficEnv.__init__() or before environment creation
from drl_vsl import create_sumocfg  # Import from original
create_sumocfg(self.effective_model_name_for_files)
```

### Flow Generation

**Original**: Called in various places
**New**: Centralized in `TrafficEnv.reset()` - ensure `skip_flow_generation` flag is properly set

## Method-by-Method Migration

### 1. State Building Methods

**Original** (in TrafficEnv):
```python
def _build_state_full_metrics(self):
    return np.array([...], dtype=np.float64)

def _build_state_minimalist(self):
    return np.zeros(3, dtype=np.float64)

def _preprocess_state(self, raw_state):
    # Normalization logic
```

**New** (in sar_framework.py):
```python
class FullMetricsState(StateRepresentation):
    def build_state(self, metrics: TrafficMetrics):
        return np.array([...], dtype=np.float64)
    
    def preprocess_state(self, raw_state):
        # Same normalization logic
```

### 2. Action Handling Methods

**Original**:
```python
def _handle_action(self, action: int):
    if self.action_strategy == "absolute_speed":
        self._action_absolute_speed(action)
    elif self.action_strategy == "relative_change":
        self._action_relative_change(action)

def _action_absolute_speed(self, action: int):
    proposed_speed_limit = self.SPEED_ACTIONS.get(action, self.current_speed_limit)
    # ... penalty calculation ...
```

**New**:
```python
class AbsoluteSpeedAction(ActionStrategy):
    def apply_action(self, action: int, current_speed_limit: float):
        proposed_speed_limit = self.speed_actions.get(action, current_speed_limit)
        # ... penalty calculation ...
        return proposed_speed_limit, penalty
```

### 3. Reward Calculation Methods

**Original**:
```python
def _calculate_reward(self):
    if self.reward_fn == "mobility":
        return self._reward_mobility_focused()
    elif self.reward_fn == "safety":
        return self._reward_safety_focused()
    elif self.reward_fn == "balanced":
        return self._reward_balanced()
```

**New**:
```python
# In TrafficEnv.step()
reward = self.reward_func.calculate(self.metrics, action_penalty, collision_penalty)
```

## Configuration Changes

### Normalization Bounds

**Original**: Loaded in `_load_or_set_normalization_bounds()`
**New**: Passed via `sar_config` dictionary

```python
# Original
self._load_or_set_normalization_bounds(normalization_bounds_path)

# New approach
sar_config = {
    'max_flow': bounds_data.get("max_flow", 10000.0),
    'max_occupancy': bounds_data.get("max_occupancy", 100.0),
    'max_queue_length': bounds_data.get("max_queue_length", 575.0 * 3 / 7)
}
```

## Common Migration Pitfalls

### 1. Missing Imports

```python
# Don't forget these imports in files using the new framework
from sar_framework import (
    create_state_representation,
    create_action_strategy,
    create_reward_function,
    TrafficMetrics
)
```

### 2. State Variable Access

**Original**: Direct attribute access
```python
self.flow_upstream = 0
self.flow_downstream = 0
```

**New**: Through metrics object
```python
self.metrics.flow_upstream = 0
self.metrics.flow_downstream = 0
```

### 3. Time Tracking for State

**Original**: `self.time_since_last_action`
**New**: Must be tracked in state representation object if needed

```python
# In FullMetricsState
if hasattr(self.state_repr, 'time_since_last_action'):
    self.state_repr.time_since_last_action += 1
```

### 4. Historical Data Access

**Original**: Direct deque attributes
```python
self.flow_downstream_history.append(flow)
```

**New**: Through metrics object
```python
self.metrics.flow_downstream_history.append(flow)
```

## Validation Strategies

### 1. Output Comparison Test

```python
# Create environments with same parameters
old_env = OldTrafficEnv(...)  # Original
new_env = TrafficEnvCompat(...)  # New with same string params

# Run same actions
for action in [0, 2, 4, 2, 0]:
    obs_old, reward_old, done_old, _, info_old = old_env.step(action)
    obs_new, reward_new, done_new, _, info_new = new_env.step(action)
    
    # Compare
    assert np.allclose(obs_old, obs_new), f"Observations differ"
    assert abs(reward_old - reward_new) < 0.001, f"Rewards differ"
```

### 2. Model Loading Test

```python
# Ensure models trained with old code work with new
model = DQN.load("old_model.zip")

# Should work with compatibility wrapper
env = TrafficEnvCompat(
    state_representation="full_metrics",
    action_strategy="absolute_speed",
    reward_fn="balanced",
    # ... other params matching training
)

obs = env.reset()
action, _ = model.predict(obs)  # Should not crash
```

### 3. Performance Benchmark

```python
import time

# Benchmark step performance
n_steps = 1000
start = time.time()
for _ in range(n_steps):
    env.step(env.action_space.sample())
elapsed = time.time() - start

print(f"Steps per second: {n_steps / elapsed:.2f}")
# Should be within 5% of original performance
```

## Hyperparameter Tuning Migration

### Update the objective function:

```python
def objective(trial, ...):
    # Old
    env = TrafficEnv(
        action_strategy="absolute_speed",
        state_representation="full_metrics",
        reward_fn=reward_fn,
        ...
    )
    
    # New
    sar_config = load_bounds_for_tuning()
    env = TrafficEnvCompat(
        state_representation="full_metrics",
        action_strategy="absolute_speed",
        reward_fn=reward_fn,
        sar_config=sar_config,
        ...
    )
```

## Gradual Migration Strategy

### Phase 1: Use Compatibility Layer (Week 1)
- Replace `TrafficEnv` imports with `TrafficEnvCompat`
- No other code changes needed
- Validate identical behavior

### Phase 2: Update Training Scripts (Week 2)
- Modify `train_model()` to use new constructors
- Update `test_model()` similarly
- Keep original files as backup

### Phase 3: Update Tuning Scripts (Week 3)
- Migrate `drl_vsl_hp-tune.py`
- Test with small tuning runs first
- Validate hyperparameter passing

### Phase 4: Custom Components (Week 4)
- Create new SAR components
- Start A/B testing
- Document successful configurations

## Rollback Plan

If issues arise:

1. **Immediate Rollback**: 
   ```python
   # Rename files
   mv drl_vsl.py.bak drl_vsl.py
   mv drl_vsl_hp-tune.py.bak drl_vsl_hp-tune.py
   ```

2. **Partial Rollback**: Use compatibility layer indefinitely
   ```python
   # Just use TrafficEnvCompat everywhere
   # It accepts string parameters like the original
   ```

3. **Debug Mode**: Add logging to compare old vs new
   ```python
   # In step() method
   logger.debug(f"Old reward calc: {old_method()}")
   logger.debug(f"New reward calc: {new_method()}")
   ```

## Testing Checklist

- [ ] All unit tests pass
- [ ] Can load and run existing trained models
- [ ] Training curves match (within 5% variance)
- [ ] Evaluation metrics consistent
- [ ] No performance degradation (< 5% slower)
- [ ] Hyperparameter tuning works
- [ ] All reward functions produce same values
- [ ] State normalization identical
- [ ] Action penalties match

## Known Issues

1. **Custom Callbacks**: May need updates if they access internal state
2. **Logging**: TrafficDataLogger references might need updates
3. **Multiprocessing**: Ensure all SAR components are picklable

## Support Resources

- Original code backup: `legacy/drl_vsl.py`
- Comparison notebook: `notebooks/migration_validation.ipynb`
- Test suite: `tests/test_migration.py`

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2024-01-XX | 1.0 | Initial migration |
| TBD | 1.1 | Hyperparameter tuning support |
| TBD | 1.2 | Custom component registry |

## Final Notes

The migration is designed to be incremental and reversible. Start with the compatibility layer and gradually move to direct component usage as confidence grows. Always maintain backups and validate behavior at each step.

Remember: The goal is to enable easier experimentation, not to change existing behavior. If results differ significantly, investigate before proceeding.