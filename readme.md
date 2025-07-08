# DRL-VSL System Architecture and Design Documentation

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Architecture Overview](#architecture-overview)
4. [Core Design Patterns](#core-design-patterns)
5. [Component Architecture](#component-architecture)
6. [Module Descriptions](#module-descriptions)
7. [Data Flow and Interactions](#data-flow-and-interactions)
8. [Configuration System](#configuration-system)
9. [Extension Points](#extension-points)
10. [Deployment and Operations](#deployment-and-operations)
11. [Development Guidelines](#development-guidelines)

---

## 1. Executive Summary

The DRL-VSL (Deep Reinforcement Learning for Variable Speed Limit) system is a modular, extensible framework for training and deploying intelligent traffic control agents. The system uses deep reinforcement learning to optimize variable speed limits on highways, aiming to improve traffic flow, reduce congestion, and enhance safety.

### Key Features

- **Modular SAR Framework**: Pluggable State representations, Action strategies, and Reward functions
- **Multi-Algorithm Support**: Currently supports DQN with provisions for PPO, SAC, TD3
- **SUMO Integration**: Realistic traffic simulation using SUMO (Simulation of Urban Mobility)
- **Experiment Management**: Systematic evaluation and comparison of different configurations
- **Configurable Training**: YAML-based configuration for easy parameter management
- **Parallel Training**: Support for training multiple models simultaneously
- **Comprehensive Logging**: TensorBoard integration and detailed metrics tracking

### Technology Stack

- **Core Framework**: Python 3.8+
- **RL Library**: Stable Baselines3
- **Traffic Simulation**: SUMO
- **Deep Learning**: PyTorch
- **Configuration**: YAML
- **Data Analysis**: Pandas, NumPy, Matplotlib

---

## 2. System Overview

### 2.1 Problem Domain

Variable Speed Limit (VSL) systems dynamically adjust speed limits based on traffic conditions to:
- Maximize traffic throughput
- Reduce congestion and travel times
- Improve safety by harmonizing speeds
- Minimize emissions through smoother traffic flow

### 2.2 Solution Approach

The DRL-VSL system trains reinforcement learning agents that learn optimal VSL control policies through interaction with realistic traffic simulations. The modular architecture allows researchers to experiment with different:
- State representations (what the agent observes)
- Action strategies (how speed limits are set)
- Reward functions (what objectives to optimize)

### 2.3 System Capabilities

1. **Traffic Environment Simulation**
   - Integration with SUMO for realistic traffic modeling
   - Support for different traffic scenarios and demand patterns
   - Real-time metrics collection (flow, occupancy, speed, queue length)

2. **Reinforcement Learning**
   - Deep Q-Network (DQN) implementation
   - Configurable neural network architectures
   - Hyperparameter optimization support

3. **Experiment Management**
   - Systematic comparison of different SAR configurations
   - Automated experiment execution and result collection
   - Performance visualization and ranking

4. **Deployment Options**
   - Standalone training mode
   - Parallel multi-configuration training
   - Evaluation mode for testing trained models

---

## 3. Architecture Overview

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                       │
│                   (CLI, Configuration Files)                │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                      Training Pipeline                      │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐     │
│  │   Config    │  │  Experiment  │  │    Analysis     │     │
│  │   Loader    │  │    Runner    │  │   Comparator    │     │
│  └─────────────┘  └──────────────┘  └─────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                        Core Framework                       │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│  │    State     │  │    Action    │  │     Reward      │    │
│  │Representation│  │   Strategy   │  │    Function     │    │
│  └──────────────┘  └──────────────┘  └─────────────────┘    │
│                                                             │
│  ┌────────────────────────────────────────────────────┐     │
│  │               Traffic Environment                  │     │
│  │  (TrafficEnv, SUMO Integration, Metrics)           │     │
│  └────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────┐
│                    External Dependencies                    │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐     │
│  │    SUMO     │  │Stable Base-  │  │    PyTorch      │     │
│  │ Simulator   │  │  lines3      │  │                 │     │
│  └─────────────┘  └──────────────┘  └─────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Component Hierarchy

```
drl-vsl-project/
├── core/                      # Core framework components
│   ├── __init__.py           # Package initialization and exports
│   ├── sar_framework.py      # SAR base classes and implementations
│   ├── drl_vsl.py           # Traffic environment implementation
│   └── drl_vsl_integration.py # Integration utilities
│
├── sar_components/           # Extended SAR implementations
│   ├── states/              # Additional state representations
│   ├── actions/             # Additional action strategies
│   └── rewards/             # Additional reward functions
│
├── experiments/             # Experiment management
│   ├── run_experiments.py   # Experiment runner
│   ├── compare_sar.py      # Results comparison tools
│   └── configs/            # Experiment configurations
│
├── training/               # Training pipeline
│   ├── drl_vsl_train.py   # Main training script
│   └── config/            # Training configurations
│
└── helpers/               # Utility scripts
    ├── check_integration.py
    ├── fix_imports.py
    └── setup_sumo_config.py
```

---

## 4. Core Design Patterns

### 4.1 Strategy Pattern (SAR Framework)

The core of the system uses the Strategy pattern to allow pluggable implementations of:

```python
# Abstract Strategy interfaces
class StateRepresentation(ABC):
    """Defines how environment state is observed"""
    
class ActionStrategy(ABC):
    """Defines how actions are interpreted"""
    
class RewardFunction(ABC):
    """Defines optimization objectives"""
```

This allows researchers to mix and match different strategies without modifying core code.

### 4.2 Factory Pattern

Factory functions create appropriate SAR components based on string identifiers:

```python
def create_state_representation(name: str, config: Dict) -> StateRepresentation
def create_action_strategy(name: str, config: Dict) -> ActionStrategy
def create_reward_function(name: str, config: Dict) -> RewardFunction
```

### 4.3 Template Method Pattern

The `TrafficEnv` class defines the skeleton of the simulation loop, with SAR components filling in specific behaviors:

```python
class TrafficEnv(gym.Env):
    def step(self, action):
        # Template method
        new_speed = self.action_strategy.apply_action(action, current_speed)
        # ... simulation steps ...
        reward = self.reward_function.calculate(metrics)
        observation = self.state_representation.get_observation(metrics)
        return observation, reward, done, info
```

### 4.4 Observer Pattern

The logging and metrics collection system observes training progress:

```python
class TrafficDataLogger:
    """Observes and logs simulation metrics"""
    
class EvalCallback:
    """Observes training progress for evaluation"""
```

### 4.5 Configuration Pattern

YAML-based configuration with validation and defaults:

```python
class Config:
    """Loads, validates, and provides access to configuration"""
    def __init__(self, config_path: str)
    def get(self, key_path: str, default=None)
```

---

## 5. Component Architecture

### 5.1 Core Components

#### 5.1.1 SAR Framework (`core/sar_framework.py`)

**Purpose**: Defines the modular architecture for State-Action-Reward components.

**Key Classes**:
- `TrafficMetrics`: Data container for all traffic measurements
- `StateRepresentation`: Abstract base for state representations
- `ActionStrategy`: Abstract base for action strategies  
- `RewardFunction`: Abstract base for reward functions

**Concrete Implementations**:
- States: `FullMetricsState`, `MinimalState`
- Actions: `AbsoluteSpeedAction`, `RelativeSpeedAction`
- Rewards: `MobilityReward`, `SafetyReward`, `BalancedReward`

#### 5.1.2 Traffic Environment (`core/drl_vsl.py`)

**Purpose**: Implements the Gymnasium environment interface for traffic simulation.

**Key Features**:
- SUMO integration for realistic traffic simulation
- Metrics collection and aggregation
- VSL enforcement modes (recommend, all_vehicles, cavs_only)
- Episode management and reset logic

**Main Class**:
```python
class TrafficEnv(gym.Env):
    def __init__(self, port, model_name, state_representation, 
                 action_strategy, reward_function, ...)
    def step(self, action) -> Tuple[observation, reward, done, info]
    def reset(self) -> Tuple[observation, info]
```

#### 5.1.3 Integration Utilities (`core/drl_vsl_integration.py`)

**Purpose**: Helper functions for creating and configuring environments.

**Key Functions**:
- `create_traffic_env_from_config()`: Factory for environment creation
- `create_train_env_helper()`: Simplified training environment creation
- `create_eval_env_helper()`: Simplified evaluation environment creation
- `load_sar_config_from_file()`: Configuration loading utilities

### 5.2 SAR Components

#### 5.2.1 State Representations (`sar_components/states/`)

**Extended States**:
- `MARVELState`: Includes downstream agent's action for coordinated control
- `DensityFocusedState`: Traffic density-centric representation
- `MultiSegmentDensityState`: Spatial awareness across road segments
- `QueueLengthState`: Queue-focused for congestion management
- `QueueDynamicsState`: Advanced queue analysis with shockwave estimation

#### 5.2.2 Action Strategies (`sar_components/actions/`)

**Extended Actions**:
- `MARVELSpeedAction`: 5-level speed control (30-70 mph)
- `TrafficAdaptiveAction`: Adapts action space to traffic conditions
- `OccupancyBasedAction`: Direct occupancy-to-speed mapping
- `GradualChangeAction`: Smooth transitions for comfort
- `MomentumBasedAction`: Physics-inspired smooth control

#### 5.2.3 Reward Functions (`sar_components/rewards/`)

**Extended Rewards**:
- `MARVELReward`: Three-component reward (adaptability, safety, mobility)
- `EmissionReward`: Environmental impact optimization
- `FuelEfficiencyReward`: Fuel consumption minimization
- `PassengerComfortReward`: Ride quality optimization
- `SmoothFlowReward`: Traffic harmonization

### 5.3 Training Pipeline

#### 5.3.1 Main Training Script (`training/drl_vsl_train.py`)

**Purpose**: Orchestrates the complete training pipeline.

**Key Features**:
- YAML configuration loading and validation
- Hyperparameter management (manual + Optuna integration)
- Parallel training support
- Model checkpointing and evaluation
- TensorBoard integration

**Training Modes**:
1. Single configuration training
2. Selected combinations training
3. All custom combinations
4. All reward functions with base state/action

#### 5.3.2 Configuration System (`training/config/`)

**Configuration Structure**:
```yaml
sar_config:
  state_representation: "full_metrics"
  action_strategy: "absolute_speed"
  reward_function: "balanced"
  custom_combinations: [...]

model:
  algorithm: "DQN"
  vsl_enforcement: "recommend"

training:
  total_timesteps: 100000
  num_train_envs: 4
  
hyperparameters:
  use_optuna: false
  defaults: {...}
  
execution:
  training_mode: "selected"
  selected_combinations: "marvel_pure"
```

### 5.4 Experiment Management

#### 5.4.1 Experiment Runner (`experiments/run_experiments.py`)

**Purpose**: Systematic execution of experiments with different SAR configurations.

**Key Features**:
- Command-line interface for experiment control
- Batch experiment execution
- Configuration file support
- Progress tracking and logging

#### 5.4.2 Results Comparator (`experiments/compare_sar.py`)

**Purpose**: Analysis and comparison of experiment results.

**Key Features**:
- Results loading and aggregation
- Performance ranking based on multiple criteria
- Visualization (plots, heatmaps, comparisons)
- Report generation
- CSV export for further analysis

**Main Class**:
```python
class SARComparator:
    def load_results(self, experiment_names)
    def rank_configurations(self, criteria, weights)
    def plot_performance_comparison(self, metrics)
    def generate_report(self, output_path)
```

---

## 6. Module Descriptions

### 6.1 Core Module (`core/`)

**Purpose**: Foundation of the DRL-VSL system providing base functionality.

**Components**:
- **sar_framework.py**: Core SAR architecture and base implementations
- **drl_vsl.py**: Traffic environment and SUMO integration
- **drl_vsl_integration.py**: Helper functions and utilities
- **__init__.py**: Package exports and version management

**Key Abstractions**:
```python
# Traffic metrics container
@dataclass
class TrafficMetrics:
    avg_speed_before: float
    flow_upstream: float
    flow_downstream: float
    queue_length_upstream: float
    occupancy_upstream: float
    # ... additional metrics
```

### 6.2 SAR Components Module (`sar_components/`)

**Purpose**: Extended implementations of SAR components for specialized scenarios.

**Organization**:
```
sar_components/
├── states/          # State representations
│   ├── marvel_states.py      # MARVEL-specific states
│   ├── density_states.py     # Density-focused states
│   └── queue_states.py       # Queue-focused states
├── actions/         # Action strategies
│   ├── marvel_actions.py     # MARVEL actions
│   ├── adaptive_actions.py   # Adaptive strategies
│   └── smooth_actions.py     # Smooth control
└── rewards/         # Reward functions
    ├── marvel_rewards.py     # MARVEL rewards
    ├── emission_rewards.py   # Environmental objectives
    └── comfort_rewards.py    # Comfort objectives
```

### 6.3 Experiments Module (`experiments/`)

**Purpose**: Tools for systematic experimentation and comparison.

**Components**:
- **run_experiments.py**: Experiment execution engine
- **compare_sar.py**: Results analysis and comparison
- **configs/**: Pre-defined experiment configurations
  - mobility_configs.yaml: Flow optimization experiments
  - safety_configs.yaml: Safety-focused experiments
  - custom_configs.yaml: User-defined experiments

### 6.4 Training Module (`training/`)

**Purpose**: Main training pipeline and configuration.

**Components**:
- **drl_vsl_train.py**: Training orchestration
- **config/drl_vsl_train_config.yaml**: Master configuration file

### 6.5 Helpers Module (`helpers/`)

**Purpose**: Utility scripts for setup and maintenance.

**Scripts**:
- **check_integration.py**: Verify system integration
- **test_marvel.py**: Test MARVEL components
- **setup_sumo_config.py**: Initialize SUMO configurations
- **fix_experiments_setup.py**: Fix experiment directory structure

---

## 7. Data Flow and Interactions

### 7.1 Training Data Flow

```
1. Configuration Loading
   YAML File → Config Object → Validation
   
2. Environment Creation
   Config → SAR Components → TrafficEnv → SUMO
   
3. Training Loop
   Agent → Action → Environment → State/Reward → Agent
   
4. Metrics Collection
   Environment → Logger → TensorBoard/CSV
   
5. Model Persistence
   Agent → Checkpoints → Disk Storage
```

### 7.2 Component Interactions

```python
# 1. Environment receives action from agent
action = agent.predict(observation)

# 2. Action strategy interprets action
new_speed_limit = action_strategy.apply_action(action, current_speed)

# 3. Environment applies speed limit to SUMO
traci.lane.setMaxSpeed(lane_id, new_speed_limit)

# 4. SUMO simulates traffic
traci.simulationStep()

# 5. Environment collects metrics
metrics = TrafficMetrics(
    flow_upstream=traci.edge.getLastStepVehicleNumber(...),
    occupancy_upstream=traci.inductionloop.getOccupancy(...),
    # ...
)

# 6. Reward function calculates reward
reward = reward_function.calculate(metrics)

# 7. State representation builds observation
observation = state_representation.get_observation(metrics)

# 8. Return to agent
return observation, reward, done, info
```

### 7.3 Experiment Workflow

```
1. Configuration Selection
   User → CLI → Config Files → Experiment List
   
2. Parallel Execution
   Experiment List → Process Pool → Multiple Training Instances
   
3. Results Collection
   Training → Results JSON → Experiment Results Directory
   
4. Analysis
   Results Directory → Comparator → Rankings/Plots/Reports
```

---

## 8. Configuration System

### 8.1 Configuration Hierarchy

```yaml
# Master configuration structure
sar_config:           # SAR framework settings
  ├── state_representation
  ├── action_strategy
  ├── reward_function
  └── custom_combinations[]
  
model:               # Model-specific settings
  ├── algorithm
  └── vsl_enforcement
  
training:            # Training parameters
  ├── total_timesteps
  ├── num_train_envs
  └── eval_freq
  
hyperparameters:     # Algorithm hyperparameters
  ├── use_optuna
  ├── defaults
  └── overrides
  
execution:           # Execution options
  ├── training_mode
  ├── parallel
  └── selected_combinations
  
environment:         # Environment settings
  ├── ports
  ├── sumo_binary
  └── normalization
  
sumo:               # SUMO-specific configuration
  ├── config_mode
  └── custom_config
```

### 8.2 Configuration Loading Process

1. **File Loading**: YAML file parsed into dictionary
2. **Validation**: Structure and value validation
3. **Default Application**: Missing values filled with defaults
4. **Override Application**: Manual overrides applied
5. **Access**: Dot notation access (e.g., `config.get('training.timesteps')`)

### 8.3 Custom Combinations

Define specific SAR configurations for training:

```yaml
custom_combinations:
  - name: "high_performance"
    state: "full_metrics"
    action: "absolute_speed"
    reward: "mobility"
    
  - name: "safety_first"
    state: "queue_dynamics"
    action: "gradual_change"
    reward: "safety"
```

---

## 9. Extension Points

### 9.1 Adding New State Representations

1. Create new class inheriting from `StateRepresentation`
2. Implement required methods:
   ```python
   class CustomState(StateRepresentation):
       def _setup(self): ...
       def get_observation_space(self): ...
       def build_state(self, metrics): ...
       def preprocess_state(self, raw_state): ...
   ```
3. Register in factory function
4. Add to valid states in configuration

### 9.2 Adding New Action Strategies

1. Create new class inheriting from `ActionStrategy`
2. Implement required methods:
   ```python
   class CustomAction(ActionStrategy):
       def _setup(self): ...
       def get_action_space(self): ...
       def apply_action(self, action, current_speed): ...
   ```
3. Register in factory function
4. Add to valid actions in configuration

### 9.3 Adding New Reward Functions

1. Create new class inheriting from `RewardFunction`
2. Implement calculation method:
   ```python
   class CustomReward(RewardFunction):
       def calculate(self, metrics, action_penalty, collision_penalty): ...
   ```
3. Register in factory function
4. Add to valid rewards in configuration

### 9.4 Adding New RL Algorithms

1. Extend training script to support new algorithm
2. Add algorithm-specific hyperparameters
3. Update model creation logic
4. Add to valid algorithms in configuration

---

## 10. Deployment and Operations

### 10.1 System Requirements

**Hardware**:
- CPU: 4+ cores recommended for parallel training
- RAM: 8GB minimum, 16GB+ recommended
- GPU: Optional but beneficial for large networks
- Storage: 10GB+ for logs and models

**Software**:
- Python 3.8+
- SUMO 1.8.0+
- CUDA (optional for GPU acceleration)

### 10.2 Installation Process

```bash
# 1. Clone repository
git clone <repository-url>
cd drl-vsl-project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set SUMO_HOME
export SUMO_HOME=/path/to/sumo

# 5. Verify installation
python helpers/check_integration.py
```

### 10.3 Training Workflow

```bash
# 1. Configure experiment
# Edit training/config/drl_vsl_train_config.yaml

# 2. Validate configuration
python training/drl_vsl_train.py --validate-only

# 3. Run training
python training/drl_vsl_train.py

# 4. Monitor progress
tensorboard --logdir logs/

# 5. Compare results
python experiments/compare_sar.py --action report
```

### 10.4 Model Deployment

Trained models can be deployed in several ways:

1. **Evaluation Mode**: Test on new scenarios
2. **Real-time Control**: Interface with actual traffic systems
3. **Simulation Studies**: Analyze performance in various conditions

### 10.5 Monitoring and Logging

**Log Locations**:
- Training logs: `logs/<model_name>/`
- Model checkpoints: `rl_models/<model_name>/`
- SUMO logs: `logs/sumo_log/`
- Experiment results: `experiments/experiment_results/`

**Metrics Tracked**:
- Training: Loss, reward, exploration rate
- Traffic: Flow, speed, occupancy, queue length
- Safety: Collisions, speed variance
- Performance: Episode length, convergence

---

## 11. Development Guidelines

### 11.1 Code Organization

**Principles**:
- Single Responsibility: Each class/function has one clear purpose
- Dependency Injection: Components receive dependencies as parameters
- Interface Segregation: Small, focused interfaces over large ones
- Open/Closed: Open for extension, closed for modification

**File Structure**:
```python
# Module docstring
"""
Module description and purpose
"""

# Imports (grouped and ordered)
import standard_library
import third_party
from local import modules

# Constants
CONSTANT_VALUE = 42

# Classes
class MainClass:
    """Class docstring"""
    
# Functions
def utility_function():
    """Function docstring"""
    
# Main guard
if __name__ == "__main__":
    main()
```

### 11.2 Testing Strategy

**Unit Tests**: Test individual components
```python
def test_state_representation():
    state = FullMetricsState(config)
    metrics = TrafficMetrics(...)
    observation = state.get_observation(metrics)
    assert observation.shape == (9,)
```

**Integration Tests**: Test component interactions
```python
def test_environment_step():
    env = TrafficEnv(...)
    obs, reward, done, info = env.step(action=0)
    assert isinstance(reward, float)
```

**System Tests**: End-to-end training runs
```bash
python training/drl_vsl_train.py --config test_config.yaml
```

### 11.3 Documentation Standards

**Code Documentation**:
- Module-level docstrings explaining purpose
- Class docstrings with attributes
- Function docstrings with parameters/returns
- Inline comments for complex logic

**User Documentation**:
- README files for each module
- Configuration examples
- Tutorial notebooks
- API reference

### 11.4 Version Control

**Branch Strategy**:
- `main`: Stable releases
- `develop`: Integration branch
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `experiment/*`: Research experiments

**Commit Messages**:
```
type(scope): subject

body (optional)

footer (optional)
```

Types: feat, fix, docs, style, refactor, test, chore

### 11.5 Performance Optimization

**Training Performance**:
- Use parallel environments
- Optimize hyperparameters with Optuna
- Profile code to identify bottlenecks
- Use GPU when available

**Simulation Performance**:
- Minimize SUMO calls
- Batch metrics collection
- Use appropriate step lengths
- Clean up temporary files

### 11.6 Debugging Tips

**Common Issues**:

1. **SUMO Connection Errors**
   - Check SUMO_HOME environment variable
   - Verify port availability
   - Check SUMO binary path

2. **Memory Issues**
   - Reduce number of parallel environments
   - Decrease replay buffer size
   - Monitor system resources

3. **Training Instability**
   - Check reward scaling
   - Verify state normalization
   - Adjust learning rate
   - Increase exploration

**Debug Tools**:
```python
# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Print shapes and values
logger.debug(f"Observation shape: {obs.shape}")
logger.debug(f"Reward: {reward}")

# Visualize with SUMO GUI
sumo_binary: 'sumo-gui'
```

---

## Appendix A: Glossary

- **CAV**: Connected Autonomous Vehicle
- **DQN**: Deep Q-Network
- **DRL**: Deep Reinforcement Learning
- **SAR**: State-Action-Reward framework
- **SUMO**: Simulation of Urban Mobility
- **VSL**: Variable Speed Limit

## Appendix B: Configuration Reference

[See training/config/drl_vsl_train_config.yaml for complete reference]

## Appendix C: API Reference

[Generated from docstrings using automated tools]