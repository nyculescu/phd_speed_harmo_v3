from traffic_environment.rl_gym_environments import MPCBasedTrafficEnv

if __name__ == '__main__':
    # Initialize environment with default PID parameters
    env = MPCBasedTrafficEnv(9000, "PID_BASED", 0, is_learning=True, base_gen_car_distrib=["uniform", 2750])
    # Automatically tune PID parameters over multiple episodes
    result = env.tune_pid_parameters(num_episodes=5)

    # Save optimized parameters for reuse later
    env.save_pid_parameters('optimized_pid_params.npy')

    print("Optimized Parameters:")
    print(f"Kp: {result.x[0]:.3f}, Ki: {result.x[1]:.3f}, Kd: {result.x[2]:.3f}")
