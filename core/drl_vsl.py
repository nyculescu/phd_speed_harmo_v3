# drl_vsl_refactored.py
"""
Refactored Traffic Environment with Modular SAR Framework

This module contains the refactored TrafficEnv class that uses the modular
State-Action-Reward framework components.
"""

import logging
import os
import time
import traci
from traci import FatalTraCIError, TraCIException
import subprocess
import numpy as np
import gymnasium as gym
from typing import Optional, Dict, Any, Tuple, Union
from pathlib import Path
from collections import deque
import psutil
from datetime import datetime

# Import the SAR framework components
from core.sar_framework import (
    StateRepresentation, ActionStrategy, RewardFunction,
    TrafficMetrics, create_state_representation, 
    create_action_strategy, create_reward_function
)

# Import SUMO configuration
try:
    from traffic_environment.sumo_config import SumoConfig, load_sumo_config, get_preset_config
except ImportError:
    # Fallback if sumo_config module doesn't exist yet
    SumoConfig = None
    load_sumo_config = None
    get_preset_config = None

logger = logging.getLogger(__name__)

# Keep all the SUMO configuration constants
edges = ["seg_10_before","seg_9_before","seg_8_before","seg_7_before","seg_6_before","seg_5_before","seg_4_before","seg_3_before","seg_2_before","seg_1_before","seg_0_before","seg_0_after","seg_1_after"]
seg_1_before = ["seg_1_before_2", "seg_1_before_1", "seg_1_before_0"]
seg_0_before = ["seg_0_before_2", "seg_0_before_1", "seg_0_before_0"]
segments_before = [["seg_10_before_2", "seg_10_before_1", "seg_10_before_0"],
                   ["seg_9_before_2", "seg_9_before_1", "seg_9_before_0"],
                   ["seg_8_before_2", "seg_8_before_1", "seg_8_before_0"],
                   ["seg_7_before_2", "seg_7_before_1", "seg_7_before_0"],
                   ["seg_6_before_2", "seg_6_before_1", "seg_6_before_0"],
                   ["seg_5_before_2", "seg_5_before_1", "seg_5_before_0"],
                   ["seg_4_before_2", "seg_4_before_1", "seg_4_before_0"],
                   ["seg_3_before_2", "seg_3_before_1", "seg_3_before_0"],
                   ["seg_2_before_2", "seg_2_before_1", "seg_2_before_0"],
                   ["seg_1_before_2", "seg_1_before_1", "seg_1_before_0"],
                   ["seg_0_before_2", "seg_0_before_1", "seg_0_before_0"]]
loops_before = [["loop_seg_0_before_2A", "loop_seg_0_before_1A", "loop_seg_0_before_0A"],
                ["loop_seg_0_before_2B", "loop_seg_0_before_1B", "loop_seg_0_before_0B"],
                ["loop_seg_0_before_2C", "loop_seg_0_before_1C", "loop_seg_0_before_0C"],
                ["loop_seg_0_before_2D", "loop_seg_0_before_1D", "loop_seg_0_before_0D"]]

SUMO_CFG_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
    <configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
        <input>
            <net-file value="../3_2_merge.net.xml"/>
            <route-files value="../generated_flows/generated_flows_{file_postfix}.rou.xml"/>
            <additional-files value="../loops_detectors.add.xml"/>
            <gui-settings-file value="../colored.view.xml"/>
        </input>
        <processing>
            <lateral-resolution value="0.2"/>
        </processing>
    </configuration>
    """

def create_sumocfg(file_postfix):
    output_dir = os.path.abspath(os.path.join("traffic_environment", "sumo", "generated_configs"))
    os.makedirs(output_dir, exist_ok=True)

    filename = f"3_2_merge_{file_postfix}.sumocfg"
    filepath = os.path.join(output_dir, filename)

    # Format the template
    content = SUMO_CFG_TEMPLATE.format(file_postfix=file_postfix)

    with open(filepath, 'w') as file:
        file.write(content)
    
    logger.debug(f"Created {filepath}")

class TrafficEnv(gym.Env):
    """
    Refactored Traffic Environment with modular SAR components and configurable SUMO.
    
    This environment now delegates state representation, action handling,
    and reward calculation to pluggable components, and uses configurable SUMO parameters.
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, 
                 port: int,
                 model_name: str,
                 model_idx: int,
                 sim_length: int,
                 base_gen_car_distrib: list,
                 num_of_episodes: int,
                 state_representation: Optional[StateRepresentation] = None,
                 action_strategy: Optional[ActionStrategy] = None,
                 reward_function: Optional[RewardFunction] = None,
                 vsl_enforcement: str = "recommend",
                 sumo_binary_path_override: Optional[str] = None,
                 sar_config: Optional[Dict[str, Any]] = None,
                 sumo_config: Optional[Union[str, Path, SumoConfig, Dict[str, Any]]] = None,
                 sumo_preset: Optional[str] = None):
        
        super(TrafficEnv, self).__init__()
        
        # Basic environment parameters
        self.port = port
        self.sim_length = sim_length
        self.model_name = model_name
        self.model_idx = model_idx
        self.effective_model_name_for_files = f"{model_name}_{model_idx}"
        self.sumo_binary_path_override = sumo_binary_path_override
        self.skip_flow_generation = False
        self.aggregation_time = 60
        self.sumo_process = None
        self.sumo_max_retries = 3
        self.is_sumo_initialized = False
        self.collisions = []
        self.gen_car_distrib = base_gen_car_distrib
        self.logger = TrafficDataLogger(model_name=model_name, 
                                      log_dir=Path(f"./logs/{model_name}"))
        self.num_of_episodes = num_of_episodes
        self.vsl_enforcement = vsl_enforcement
        self.default_speed_limit = 130
        
        # Initialize SAR configuration
        if sar_config is None:
            sar_config = {
                'max_flow': 10000.0,
                'max_occupancy': 100.0,
                'max_queue_length': 575.0 * 3 / 7
            }
        
        # Initialize SAR components
        self.state_repr = state_representation or create_state_representation('full_metrics', sar_config)
        self.action_strat = action_strategy or create_action_strategy('absolute_speed', sar_config)
        self.reward_func = reward_function or create_reward_function('balanced', sar_config)
        
        # Set up spaces
        self.observation_space = self.state_repr.get_observation_space()
        self.action_space = self.action_strat.get_action_space()
        
        # Initialize traffic metrics
        self.metrics = TrafficMetrics()
        
        # Initialize SUMO configuration
        self._init_sumo_config(sumo_config, sumo_preset)
        
        # Get step length from SUMO config
        self.sumo_step_length = self.sumo_config.config['sumo'].get('step_length', 1.0)
        
        # Training scenarios
        self.training_scenarios = [
            {"demand": 2500, "pattern": "uniform"},
            {"demand": 3000, "pattern": "uniform"},
            {"demand": 3500, "pattern": "uniform"},
            {"demand": 4000, "pattern": "uniform"},
            {"demand": 4500, "pattern": "uniform"},
            {"demand": 5000, "pattern": "uniform"},
        ]
        
        # Tracking variables
        self.reward_window = deque(maxlen=50)
        self.reward_threshold = -5
        self.simulation_step = 0
        self.veh_passed_downstream = 0
        
        # SUMO context
        self._sumo_start_context_prefix = ""
        self._default_sumo_binary_for_env = self.sumo_config.get_sumo_binary(self.sumo_binary_path_override)
        self._sumo_retry_sleep_func = lambda attempt, max_retries: max_retries + attempt
    
    def _init_sumo_config(self, 
                         sumo_config: Optional[Union[str, Path, SumoConfig, Dict[str, Any]]] = None,
                         sumo_preset: Optional[str] = None):
        """Initialize SUMO configuration."""
        if SumoConfig is None:
            # Fallback to old behavior if sumo_config module not available
            logger.warning("SUMO config module not available, using legacy configuration")
            self.sumo_config = None
            return
        
        # Determine configuration source
        if sumo_preset:
            # Use preset configuration
            self.sumo_config = get_preset_config(sumo_preset)
            logger.info(f"Using SUMO preset configuration: {sumo_preset}")
        elif isinstance(sumo_config, SumoConfig):
            # Already a SumoConfig instance
            self.sumo_config = sumo_config
        elif isinstance(sumo_config, dict):
            # Create from dictionary
            from traffic_environment.sumo_config import create_sumo_config_from_dict
            self.sumo_config = create_sumo_config_from_dict(sumo_config)
        elif isinstance(sumo_config, (str, Path)):
            # Load from file
            self.sumo_config = load_sumo_config(sumo_config)
        else:
            # Use default configuration
            from traffic_environment.sumo_config import get_default_sumo_config
            self.sumo_config = get_default_sumo_config()
            logger.info("Using default SUMO configuration")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment using modular components."""
        
        if isinstance(action, np.ndarray):
            action = action.item()
        
        # Initialize SUMO if not already done
        if not self.is_sumo_initialized:
            self._start_sumo()
        
        # Check SUMO responsiveness
        try:
            current_time = traci.simulation.getTime()
        except (FatalTraCIError, TraCIException):
            logger.error("Lost connection to SUMO, restarting...")
            self.is_sumo_initialized = False
            self._start_sumo()
            current_time = traci.simulation.getTime()
        
        # Apply action using the action strategy
        new_speed_limit, action_penalty = self.action_strat.apply_action(
            action, self.metrics.current_speed_limit
        )
        self.metrics.current_speed_limit = new_speed_limit
        
        # Update time tracking for state representation
        if hasattr(self.state_repr, 'time_since_last_action'):
            if action != 2:  # Assuming action 2 is "no change"
                self.state_repr.time_since_last_action = 0
            else:
                self.state_repr.time_since_last_action += 1
        
        # Apply VSL enforcement
        self._apply_vsl_enforcement(self.metrics.current_speed_limit)
        
        # Initialize data collection variables
        flow_upstream_temp = 0
        flow_downstream_temp = 0
        queue_length_temp = 0
        mean_speeds_downstream = 0
        mean_speeds_upstream = 0
        occupancy_upstream_temp = 0
        
        # Simulation steps and data aggregation
        num_sumo_steps = int(self.aggregation_time / self.sumo_step_length)
        for step in range(num_sumo_steps):
            try:
                traci.simulationStep()
                current_time = traci.simulation.getTime()
                self.simulation_step += 1
            except (FatalTraCIError, TraCIException):
                logger.error("Lost connection during simulation steps. Terminating episode.")
                observation = self.state_repr.get_observation(self.metrics)
                return observation, -10.0, True, False, {}
            
            # Collect traffic measurements
            flow_upstream_temp += traci.edge.getLastStepVehicleNumber("seg_0_before")
            flow_downstream_temp += traci.edge.getLastStepVehicleNumber("seg_0_after")
            
            # Measure queue for critical segments
            critical_segments = [seg_1_before, seg_0_before]
            queue_length_temp += sum(
                traci.lane.getLastStepHaltingNumber(lane_id) * 7.5
                for segment in critical_segments
                for lane_id in segment
            )
            
            # Speed measurements
            mean_speeds_downstream += traci.edge.getLastStepMeanSpeed("seg_0_before")
            mean_speeds_upstream += traci.edge.getLastStepMeanSpeed("seg_0_after")
            
            # Occupancy from induction loops
            occupancy_upstream_temp += sum([
                traci.inductionloop.getLastStepOccupancy(loop_id)
                for loop_ids in loops_before
                for loop_id in loop_ids
            ]) / len([loop for loops in loops_before for loop in loops])
            
            # Collision detection
            collisions_in_step = traci.simulation.getCollidingVehiclesNumber()
            if collisions_in_step > 0:
                self.collisions.append(current_time)
        
        # Update metrics
        self.metrics.avg_speed_before = mean_speeds_downstream / self.aggregation_time
        self.metrics.flow_upstream = (flow_upstream_temp / self.aggregation_time) * 3600
        self.metrics.flow_downstream = (flow_downstream_temp / self.aggregation_time) * 3600
        self.metrics.queue_length_upstream = queue_length_temp / self.aggregation_time
        self.metrics.occupancy_upstream = min(occupancy_upstream_temp / self.aggregation_time, 100.0)
        self.metrics.simulation_time = current_time
        self.metrics.simulation_step = self.simulation_step
        
        # Update historical data
        self.metrics.flow_downstream_history.append(self.metrics.flow_downstream)
        self.metrics.occupancy_downstream_history.append(self.metrics.occupancy_upstream)
        self.metrics.speed_history.append(self.metrics.avg_speed_before)
        
        # Calculate smoothed values
        self.metrics.flow_smoothed = np.mean(list(self.metrics.flow_downstream_history)) if self.metrics.flow_downstream_history else 0
        self.metrics.occupancy_smoothed = np.mean(list(self.metrics.occupancy_downstream_history)) if self.metrics.occupancy_downstream_history else 0
        
        # Collision penalty (2-hour sliding window)
        expiration_time = current_time - (2 * 3600)
        self.collisions = [t for t in self.collisions if t > expiration_time]
        collision_penalty = -5 if len(self.collisions) > 2 else 0
        self.metrics.collisions_count = len(self.collisions)
        
        # Calculate reward using the reward function
        reward = self.reward_func.calculate(self.metrics, action_penalty, collision_penalty)
        self.reward_window.append(reward)
        
        # Get observation from state representation
        observation = self.state_repr.get_observation(self.metrics)
        
        # Check termination conditions
        done = (current_time >= self.sim_length) or \
               (traci.simulation.getMinExpectedNumber() <= 0) or \
               (len(self.reward_window) == self.reward_window.maxlen and 
                np.mean(self.reward_window) < self.reward_threshold)
        
        # Log data
        self.logger.log_step_data(
            current_time, self.metrics.current_speed_limit, self.metrics.flow_upstream,
            self.metrics.flow_downstream, self.metrics.occupancy_upstream, 
            self.metrics.queue_length_upstream, reward, action, self.metrics.avg_speed_before
        )
        
        info = {
            'flow_upstream': self.metrics.flow_upstream,
            'flow_downstream': self.metrics.flow_downstream,
            'avg_speed_before': self.metrics.avg_speed_before,
            'occupancy': self.metrics.occupancy_upstream,
            'queue_length_upstream': self.metrics.queue_length_upstream,
            'speed_limit': self.metrics.current_speed_limit,
            'collisions': len(self.collisions),
            'simulation_time': current_time,
            'simulation_step': self.simulation_step
        }
        
        self.veh_passed_downstream += flow_downstream_temp
        logger.debug(f"No. of vehicles arrived: {self.veh_passed_downstream}")
        
        return observation, reward, done, False, info

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Close existing SUMO if running
        if self.is_sumo_initialized:
            self._close_sumo("Environment reset")
        
        if not self.skip_flow_generation:
            # Import flow generation functions
            from traffic_environment.flow_gen import flow_generation_fix_num_veh, flow_generation, bimodal_distribution_24h
            
            # Randomly select a scenario
            scenario = np.random.choice(self.training_scenarios)
            self.gen_car_distrib = [scenario["pattern"], scenario["demand"]]
            logger.info(f"Resetting env. New scenario: Demand={self.gen_car_distrib[1]} veh/hr")
            
            # Generate flow file
            CAV_PERCENTAGE = 10  # This should be configurable
            if self.gen_car_distrib[0] == 'uniform':
                flow_generation_fix_num_veh(
                    self.effective_model_name_for_files,
                    self.gen_car_distrib[1],
                    self.sim_length,
                    1, 1,
                    CAV_PERCENTAGE
                )
            elif self.gen_car_distrib[0] == 'bimodal':
                flow_generation(
                    self.effective_model_name_for_files,
                    bimodal_distribution_24h(self.gen_car_distrib[1] / 1000.0),
                    self.sim_length,
                    CAV_PERCENTAGE
                )
        
        # Reset metrics
        self.metrics = TrafficMetrics(current_speed_limit=self.default_speed_limit)
        
        # Reset other state variables
        self.collisions = []
        self.simulation_step = 0
        self.is_sumo_initialized = False
        self.veh_passed_downstream = 0
        
        # Reset state representation if it has internal state
        if hasattr(self.state_repr, 'time_since_last_action'):
            self.state_repr.time_since_last_action = 0
        
        # Start fresh SUMO instance
        self._start_sumo()
        
        observation = self.state_repr.get_observation(self.metrics)
        info = {
            'flow_upstream': 0, 'flow_downstream': 0, 'occupancy': 0,
            'queue_length': 0, 'speed_limit': self.default_speed_limit,
            'collisions': 0, 'simulation_time': 0, 'simulation_step': 0
        }
        
        return observation, info

    def _apply_vsl_enforcement(self, speed_limit_kmh: float):
        """Apply Variable Speed Limit enforcement based on configured mode."""
        speed_limit_ms = speed_limit_kmh / 3.6
        
        if self.vsl_enforcement == "recommend":
            for segId in seg_1_before:
                traci.lane.setMaxSpeed(segId, speed_limit_ms)
                logger.debug(f"VSL Mode {self.vsl_enforcement}: Set lane max speed to {speed_limit_kmh} km/h")
        
        elif self.vsl_enforcement == "all_vehicles":
            for segId in seg_1_before:
                traci.lane.setMaxSpeed(segId, speed_limit_ms)
                veh_ids = traci.lane.getLastStepVehicleIDs(segId)
                for veh_id in veh_ids:
                    try:
                        traci.vehicle.setSpeed(veh_id, speed_limit_ms)
                        logger.debug(f"VSL Mode {self.vsl_enforcement}: Forced all vehicles to {speed_limit_kmh} km/h")
                    except Exception as e:
                        logger.debug(f"Could not set speed for vehicle {veh_id}: {e}")
        
        elif self.vsl_enforcement == "cavs_only":
            CAV_TYPES = ["CAV_passenger", "CAV_passenger/van", "CAV_bus", "CAV_truck", "CAV_truck/trailer"]
            for segId in seg_1_before:
                traci.lane.setMaxSpeed(segId, speed_limit_ms)
                veh_ids = traci.lane.getLastStepVehicleIDs(segId)
                for veh_id in veh_ids:
                    try:
                        veh_type = traci.vehicle.getTypeID(veh_id)
                        if veh_type in CAV_TYPES:
                            traci.vehicle.setSpeed(veh_id, speed_limit_ms)
                            logger.debug(f"VSL Mode {self.vsl_enforcement}: Forced CAVs to {speed_limit_kmh} km/h")
                    except Exception as e:
                        logger.debug(f"Could not check/set speed for vehicle {veh_id}: {e}")
        else:
            logger.warning(f"Unknown VSL enforcement mode: {self.vsl_enforcement}. Using recommend.")
            for segId in seg_1_before:
                traci.lane.setMaxSpeed(segId, speed_limit_ms)

    def _get_sumo_log_identifier(self):
        """Helper to get a consistent identifier for SUMO instance logging."""
        return f"{self._sumo_start_context_prefix}{self.effective_model_name_for_files}"
    
    def _ensure_clean_traci_state(self):
        """Ensure TraCI is in a clean state before starting SUMO."""
        try:
            if traci.isLoaded():
                logger.debug("TraCI connection found active, closing it...")
                traci.close()
        except Exception as e:
            logger.warning(f"Error while checking/closing TraCI: {e}")
        time.sleep(0.5)

    def _start_sumo(self):
        """Initialize SUMO simulation using configuration."""
        log_id = self._get_sumo_log_identifier()
        
        if self.is_sumo_initialized and self.sumo_process and psutil.pid_exists(self.sumo_process.pid):
            try:
                traci.simulation.getTime()
                return
            except (FatalTraCIError, TraCIException, ConnectionResetError, BrokenPipeError):
                logger.warning(f"SUMO process ({log_id}) exists but not responsive, restarting...")
                self.is_sumo_initialized = False
        
        if self.sumo_process and psutil.pid_exists(self.sumo_process.pid):
            self._close_sumo(f"Restarting SUMO for initialization ({log_id})")
            time.sleep(3)
        elif self.sumo_process and not psutil.pid_exists(self.sumo_process.pid):
            logger.debug(f"SUMO process handle existed for {log_id} but PID was not found. Clearing handle.")
            self.sumo_process = None
        
        self._ensure_clean_traci_state()
        
        for attempt in range(self.sumo_max_retries):
            try:
                port = self.port
                route_file = os.path.abspath(os.path.join("traffic_environment", "sumo", "generated_flows", f"generated_flows_{self.effective_model_name_for_files}.rou.xml"))

                if not os.path.exists(route_file) or os.path.getsize(route_file) == 0:
                    logger.error(f"Route file missing or empty: {route_file} on attempt {attempt + 1} for {log_id}.")
                
                # Get SUMO binary
                current_sumo_binary = self._default_sumo_binary_for_env
                
                # Create log file
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                sumo_log_file = f"./logs/sumo_log/{self.effective_model_name_for_files}_{timestamp_str}.txt"

                # Get config path
                config_path = os.path.abspath(os.path.join("traffic_environment", "sumo", "generated_configs", f"3_2_merge_{self.effective_model_name_for_files}.sumocfg"))
                
                # Build SUMO command using configuration
                if self.sumo_config:
                    sumo_cmd = self.sumo_config.get_sumo_cmd(
                        port=port,
                        sim_length=self.sim_length,
                        config_file=config_path,
                        log_file=sumo_log_file,
                        binary_override=current_sumo_binary
                    )
                else:
                    # Fallback to legacy command building
                    sumo_cmd = [
                        current_sumo_binary, "-c", config_path,
                        '--start',
                        "--default.emergencydecel=7",
                        '--random-depart-offset=3600',
                        "--remote-port", str(port),
                        f"--step-length={self.sumo_step_length}",
                        "--default.action-step-length=0.2",
                        f"--end={self.sim_length}",
                        "--no-step-log",
                        "--no-warnings",
                        "--time-to-teleport", "-1",
                        "--collision.action", "warn",
                        "--log", sumo_log_file
                    ]
                
                self.sumo_process = subprocess.Popen(sumo_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                time.sleep(0.2)
                exit_code = self.sumo_process.poll()
                if exit_code is not None:
                    out, err = self.sumo_process.communicate()
                    logger.error(f"SUMO process failed on launch with exit code {exit_code}.")
                    logger.error(f"SUMO stdout: {out.decode()}")
                    logger.error(f"SUMO stderr: {err.decode()}")
                    raise RuntimeError("SUMO failed to start. Check logs for details.")
                
                logger.debug(f"SUMO command: {' '.join(sumo_cmd)}")
                logger.debug(f"Expected simulation end time: {self.sim_length}s")
                
                logger.info(f"Attempting to connect to SUMO ({log_id}) on port {port}")
                time.sleep(0.5)
                try:
                    traci.init(port=port, numRetries=5)
                except Exception as e:
                    if self.sumo_process:
                        out, err = self.sumo_process.communicate(timeout=2)
                        logger.error(f"SUMO stderr: {err.decode()}")
                    raise e
                
                logger.info(f"Successfully connected to SUMO ({log_id}) on port {port} for {self.sim_length}s simulation")
                self.is_sumo_initialized = True
                break
                
            except (FatalTraCIError, TraCIException, ConnectionRefusedError) as e:
                logger.error(f"Attempt {attempt + 1} to start/connect SUMO ({log_id}) failed: {e}")
                self._close_sumo(f"Failed to start/connect SUMO ({log_id}) on attempt {attempt+1}")
                if attempt < self.sumo_max_retries - 1:
                    time.sleep(self._sumo_retry_sleep_func(attempt, self.sumo_max_retries))
                else:
                    logger.error(f"Max retries reached for starting SUMO ({log_id}). Raising exception.")
                    raise e

    def _close_sumo(self, reason: str):
        """Safely closes the TraCI connection and terminates the SUMO process."""
        log_id = self._get_sumo_log_identifier()
        logger.debug(f"Closing SUMO for {log_id} due to: {reason}")
        
        if traci.isLoaded():
            try:
                traci.close(wait=False)
            except Exception as e:
                logger.warning(f"Exception during traci.close() for {log_id}: {e}")
        
        if self.sumo_process:
            if self.sumo_process.poll() is None:
                try:
                    logger.debug(f"Terminating SUMO process PID {self.sumo_process.pid} for {log_id}.")
                    self.sumo_process.terminate()
                    try:
                        out, err = self.sumo_process.communicate(timeout=5)
                        if err:
                            logger.warning(f"Final SUMO stderr on close for {log_id}: {err.decode().strip()}")
                        if out:
                            logger.debug(f"Final SUMO stdout on close for {log_id}: {out.decode().strip()}")
                    except subprocess.TimeoutExpired:
                        logger.warning(f"SUMO process PID {self.sumo_process.pid} did not terminate in time, killing.")
                        self.sumo_process.kill()
                        out, err = self.sumo_process.communicate()
                        if err:
                            logger.error(f"Final SUMO stderr after kill for {log_id}: {err.decode().strip()}")
                    
                    logger.debug(f"SUMO process PID {self.sumo_process.pid} has been handled.")
                
                except Exception as e:
                    logger.error(f"Exception during SUMO process termination for {log_id}: {e}")
            else:
                out, err = self.sumo_process.communicate()
                if err:
                    logger.debug(f"SUMO process for {log_id} had already terminated. Final stderr: {err.decode().strip()}")
            
            self.sumo_process = None
        self.is_sumo_initialized = False

    def close(self):
        """Closes the environment and its SUMO instance."""
        self._close_sumo(f"env.close() called for {self._get_sumo_log_identifier()}")
    
    def update_sumo_config(self, updates: Dict[str, Any]):
        """Update SUMO configuration at runtime."""
        if self.sumo_config:
            self.sumo_config.update_config(updates)
            # Update step length if changed
            if 'sumo' in updates and 'step_length' in updates['sumo']:
                self.sumo_step_length = updates['sumo']['step_length']
            logger.info(f"Updated SUMO configuration: {updates}")
        else:
            logger.warning("Cannot update SUMO config - no config object available")
    
    def set_sumo_preset(self, preset_name: str):
        """Switch to a preset SUMO configuration."""
        if get_preset_config:
            self.sumo_config = get_preset_config(preset_name)
            self.sumo_step_length = self.sumo_config.config['sumo'].get('step_length', 1.0)
            self._default_sumo_binary_for_env = self.sumo_config.get_sumo_binary(self.sumo_binary_path_override)
            logger.info(f"Switched to SUMO preset: {preset_name}")
        else:
            logger.warning("Cannot switch preset - SUMO config module not available")

# Keep the TrafficDataLogger class as is
class TrafficDataLogger:
    """Comprehensive data logger for traffic simulation and RL training."""
    
    def __init__(self, model_name: str, log_dir: Path):
        self.model_name = model_name
        self.default_speed_limit = 130
        self.output_dir = Path(log_dir) / "traffic_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = []
        self.episode_rewards = []
        self.reset()
    
    def log_step_data(self, simulation_time, current_speed_limit, flow_upstream,
                     flow_downstream, occupancy, queue_length, reward, action, avg_speed_before):
        """Log data for a single simulation step."""
        if current_speed_limit != self.last_speed_limit:
            self.speed_limit_changes += 1
            self.last_speed_limit = current_speed_limit
        
        action_map = {0: -10, 1: -5, 2: 0, 3: 5, 4: 10}
        speed_change = action_map.get(int(action), 0)
        
        step_data = {
            'timestamp': datetime.now().isoformat(),
            'simulation_time': simulation_time,
            'step': self.step_count,
            'episode': self.episode_count,
            'current_speed_limit': current_speed_limit,
            'speed_change': speed_change,
            'action': int(action),
            'flow_upstream': flow_upstream,
            'flow_downstream': flow_downstream,
            'occupancy': occupancy,
            'queue_length': queue_length,
            'reward': reward,
            'cumulative_reward': self.total_reward + reward,
            'avg_speed_before_mps': avg_speed_before,
        }
        
        self.data.append(step_data)
        self.step_count += 1
        self.total_reward += reward
    
    def save_to_csv(self, filename: str):
        """Save logged step-by-step data to a CSV file."""
        if not self.data:
            logger.warning(f"No step data to save for {filename}")
            return
        
        filepath = self.output_dir / filename
        
        try:
            import pandas as pd
            df = pd.DataFrame(self.data)
            df.to_csv(filepath, index=False)
            logger.info(f"Traffic data log saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving step data to {filepath}: {e}")
    
    def get_summary_statistics(self) -> dict:
        """Calculates a comprehensive summary for a completed simulation run."""
        if not self.data:
            logger.warning(f"No data logged for {self.model_name}; cannot generate summary.")
            return {}
        
        import pandas as pd
        df = pd.DataFrame(self.data)
        
        total_steps = len(df)
        total_sim_time_s = df['simulation_time'].max()
        avg_flow_vph = df['flow_downstream'].mean()
        flow_stability_std_dev = df['flow_downstream'].std()
        
        avg_queue_m = df['queue_length'].mean()
        max_queue_m = df['queue_length'].max()
        avg_speed_kph = df['avg_speed_before_mps'].mean() * 3.6
        speed_variance = df['avg_speed_before_mps'].var()
        
        control_actions = df[df['speed_change'] != 0].shape[0]
        control_frequency_pct = (control_actions / total_steps) * 100 if total_steps > 0 else 0
        
        summary_stats = {
            'total_steps': total_steps,
            'total_sim_time_s': total_sim_time_s,
            'avg_flow_vph': avg_flow_vph,
            'flow_stability_std_dev': flow_stability_std_dev,
            'avg_queue_m': avg_queue_m,
            'max_queue_m': max_queue_m,
            'avg_speed_kph': avg_speed_kph,
            'speed_variance': speed_variance,
            'total_control_actions': control_actions,
            'control_frequency_pct': control_frequency_pct,
        }
        
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in summary_stats.items()}
    
    def reset(self):
        """Resets the logger for a new episode or evaluation run."""
        self.start_time = time.time()
        self.data.clear()
        self.last_speed_limit = self.default_speed_limit
        self.step_count = 0
        self.total_reward = 0.0
        self.speed_limit_changes = 0
        self.best_reward = float('-inf')
        self.episode_count = 0
        logger.debug(f"TrafficDataLogger for model {self.model_name} has been reset.")