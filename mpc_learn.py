from skopt import gp_minimize
from casadi import *
import do_mpc
from traffic_environment.rl_gym_environments import TrafficEnv
from traffic_environment.road import seg_1_before
import traci
import psutil
import numpy as np

class TrafficDynamics:
    @staticmethod
    def build_model():
        """Define the system dynamics for traffic control."""
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)

        # State variable (occupancy upstream of merging point)
        x = model.set_variable(var_type='_x', var_name='occupancy', shape=(1, 1))  # Corrected shape

        # Control variable (speed limit in m/s)
        u = model.set_variable(var_type='_u', var_name='speed_limit', shape=(1, 1))  # Corrected shape

        # System dynamics: dx/dt = -a*x + b*u
        a = 0.1  # Decay rate of occupancy
        b = 0.05  # Influence of speed on occupancy
        dxdt = -a * x + b * u

        model.set_rhs('occupancy', dxdt)
        model.setup()
        return model

class MPCTuner:
    def __init__(self, model):
        self.model = model

    def tune(self, param_space, n_calls=50):
        """Tune MPC parameters using Bayesian Optimization."""
        
        def objective_function(params):
            n_horizon, weight_occupancy, weight_speed = params
            
            # Reinitialize and configure MPC controller for each parameter set
            mpc_controller = self._configure_mpc(n_horizon, weight_occupancy, weight_speed)
            
            # Evaluate performance of the new configuration
            performance_metric = self.simulate_and_evaluate(mpc_controller)
            return performance_metric

        result = gp_minimize(
            func=objective_function,
            dimensions=param_space,
            n_calls=n_calls,
            acq_func="EI",  # Expected Improvement acquisition function
            n_random_starts=10,
            random_state=42
        )
        
        return result

    def _configure_mpc(self, n_horizon, weight_occupancy, weight_speed):
        """Reconfigure the MPC controller with new parameters."""
        from do_mpc.controller import MPC

        # Ensure the model is properly set up
        assert self.model.flags['setup'], "Model must be set up before configuring MPC."

        # Create a new MPC controller instance
        mpc = MPC(self.model)

        # Configure MPC settings
        setup_mpc = {
            'n_horizon': int(n_horizon),
            't_step': 60,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 2,
            'store_full_solution': True,
        }
        mpc.set_param(**setup_mpc)

        # Set objective function with new weights
        mterm = weight_occupancy * (self.model.x['occupancy'] - 12)**2
        lterm = weight_speed * (self.model.u['speed_limit'] - 100/3.6)**2
        mpc.set_objective(mterm=mterm, lterm=lterm)

        # Set constraints
        mpc.bounds['lower', '_u', 'speed_limit'] = 50 / 3.6
        mpc.bounds['upper', '_u', 'speed_limit'] = 130 / 3.6

        # Finalize setup
        mpc.setup()
        
        return mpc

    def simulate_and_evaluate(self, mpc_controller):
        """Run simulation and evaluate performance."""
        
        state = np.array([12])  # Initial occupancy (%)
        
        total_error = 0
        for _ in range(10):  # Simulate for a few steps
            control_action = mpc_controller.make_step(state)
            state[0] += -0.1 * state[0] + 0.05 * control_action[0]  # Update state based on dynamics
            
            error = abs(state[0] - 12)  # Deviation from target occupancy
            total_error += error
        
        return total_error / 10


class MPCController:
    def __init__(self, model):
        self.mpc = self._configure_mpc(model)

    def _configure_mpc(self, model):
        """Configure the MPC controller."""
        mpc = do_mpc.controller.MPC(model)

        setup_mpc = {
            'n_horizon': 10,  # Prediction horizon
            't_step': 60,     # Time step in seconds
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 2,
            'store_full_solution': True,
        }
        mpc.set_param(**setup_mpc)

        # Objective function: minimize deviation from target occupancy and speed changes
        target_occupancy = 12  # Target occupancy (%)
        base_speed = 100 / 3.6  # Base speed in m/s (100 km/h)

        mterm = 10 * (model.x['occupancy'] - target_occupancy)**2  # Terminal cost
        lterm = (model.u['speed_limit'] - base_speed)**2           # Stage cost

        mpc.set_objective(mterm=mterm, lterm=lterm)

        # Constraints on control input (speed limits in m/s)
        mpc.bounds['lower', '_u', 'speed_limit'] = 50 / 3.6  # Min speed limit (50 km/h)
        mpc.bounds['upper', '_u', 'speed_limit'] = 130 / 3.6 # Max speed limit (130 km/h)

        mpc.setup()
        return mpc

    def get_control_action(self, current_state):
        """Get the optimal control action for the current state."""
        return self.mpc.make_step(current_state)

class MPCBasedTrafficEnv(TrafficEnv):
    def __init__(self, port, model_idx, is_learning, base_gen_car_distrib):
        super().__init__(port=port, model=None, model_idx=model_idx,
                         is_learning=is_learning, base_gen_car_distrib=base_gen_car_distrib)
        
        self.dynamics_model = TrafficDynamics.build_model()
        self.mpc_controller = MPCController(self.dynamics_model)

    def step(self, _):
        is_sumo_running = psutil.pid_exists(self.sumo_process.pid) if self.sumo_process else False
        if not is_sumo_running:
            self.start_sumo()

        current_occupancy_percent = self.occupancy

        # Get optimal speed limit from MPC controller
        u_optimal = self.mpc_controller.get_control_action(np.array([current_occupancy_percent]))[0]
        
        target_speed_kmh = round(u_optimal * 3.6 / 5) * 5
        target_speed_kmh = min(max(target_speed_kmh, 50), 130) 

        self.speed_limit = target_speed_kmh
        
        for segment in [seg_1_before]:
            [traci.lane.setMaxSpeed(segId, self.speed_limit / 3.6) for segId in segment]

class Evaluator:
    @staticmethod
    def evaluate(env):
        total_error = []
        
        for _ in range(10): 
            obs, _, done, _, _ = env.step(None)
            
            error = abs(env.target_occupancy - env.occupancy)
            total_error.append(error)
            
            if done:
                break
        
        return np.mean(total_error), np.var(total_error) 

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
if __name__ == '__main__':
    # Build the traffic dynamics model
    dynamics_model = TrafficDynamics.build_model()

    # Initialize MPC controller
    mpc_controller = MPCController(dynamics_model)

    # Define parameter space for tuning
    param_space = [
        (5, 20),       # Prediction horizon range
        (1.0, 20.0),   # Weight on occupancy deviation
        (0.1, 5.0),    # Weight on speed change penalty
    ]

    # Initialize tuner
    mpc_tuner = MPCTuner(mpc_controller)

    # Run tuning process
    result = mpc_tuner.tune(param_space, n_calls=50)

    # Best parameters found during tuning
    best_horizon, best_weight_occupancy, best_weight_speed = result.x
    print(f"Optimized Parameters: Horizon={best_horizon}, Weight Occupancy={best_weight_occupancy}, Weight Speed={best_weight_speed}")

    mpc_controller.mpc.set_param(n_horizon=int(best_horizon))
    mpc_controller.mpc.set_objective(
        mterm=best_weight_occupancy * (dynamics_model.x['occupancy'] - 12)**2,
        lterm=best_weight_speed * (dynamics_model.u['speed_limit'] - 100/3.6)**2,
    )

    # Initialize environment with SUMO integration and MPC controller
    env = MPCBasedTrafficEnv(
        port=12345,
        model_idx=0,
        is_learning=False,
        base_gen_car_distrib=500,
    )

    # Run simulation using MPC controller
    for _ in range(1440):  # Simulate for 1440 steps (24 hours)
        obs, _, done, _, _ = env.step(None)
        if done:
            break

    print("Simulation completed.")

    mean_error, variance_error = Evaluator.evaluate(env)

    print(f"Mean Occupancy Deviation: {mean_error:.2f}")
    print(f"Variance in Occupancy Deviation: {variance_error:.2f}")

    np.save('optimized_mpc_params.npy', {
        'horizon': best_horizon,
        'weight_occupancy': best_weight_occupancy,
        'weight_speed': best_weight_speed,
    })

    #####################################################################
    # params = np.load('optimized_mpc_params.npy', allow_pickle=True).item()

    # # Apply loaded parameters to MPC controller
    # mpc_controller.mpc.set_param(n_horizon=int(params['horizon']))
    # mpc_controller.mpc.set_objective(
    #     mterm=params['weight_occupancy'] * (dynamics_model.x['occupancy'] - 12)**2,
    #     lterm=params['weight_speed'] * (dynamics_model.u['speed_limit'] - 100/3.6)**2,
    # )