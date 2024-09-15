def linear_occ_reward(x):
    if 9 < x < 13:
        return 2

    if 0 < x <= 11:
        return x/11
    elif 11 < x < 100:
        return (100-x)/89
    else:
        return 0

''' This function rewards occupancy rates around 12% the most, with decreasing rewards for both lower and higher occupancies.
    - low occupancy rates (0-12%) => increases from 0.5 at x=0 to 1 at x=12
    - medium to high occupancy rates (12-80%) => decreases from 1 at x=12 to 0 at x=80
    - very low (≤0) or very high (≥80) occupancy rates => reward is 0
    The function is continuous at x=12, where both pieces evaluate to 1. However, it is not differentiable at x=12 due to the change in slope.
    Ref.: Kidando, E., Moses, R., Ozguven, E. E., & Sando, T. (2017). Evaluating traffic congestion using the traffic occupancy and speed distribution relationship: An application of Bayesian Dirichlet process mixtures of generalized linear model. Journal of Transportation Technologies, 7(03), 318.
'''
def quad_occ_reward(occupancy):
    if 0 < occupancy <= 12:
        return ((0.5 * occupancy) + 6) / 12
    elif 12 < occupancy < 80:
        return ((occupancy-80)**2/68**2)
    else:
        return 0