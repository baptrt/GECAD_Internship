## Modelisation of the market at one time step

from Environment.ADMM_step import admm_update, update_lagrange

import numpy as np
import matplotlib.pyplot as plt
# import time
# from tabulate import tabulate

def simulate_market(T, agents, max_iters, a, b, tmin, tmax, pmin, pmax, gamma, local_prices, rho, rhol, max_error=1e-8):
    n, m = T.shape
    R = np.zeros(n)
    S = np.zeros(n)
    P = np.zeros(n)
    Mu = np.zeros(n)
    T_mean = np.zeros(n)
    errors = []
    #times = []
    error = 2 * max_error
    k = 0
    bt1 = np.zeros((n, m))
    
    n_agents = len(agents)

    while k < max_iters and error > max_error:
        #print(f"Iteration number {k}")
        k += 1

        #start_time = time.time()

        # Local update
        T_new, P, Mu, T_mean = admm_update(T, agents, max_iters, a, b, tmin, tmax, pmin, pmax, gamma, rho, rhol, bt1, max_error, P, Mu, T_mean)

        # Global update
        R, S, local_prices, bt1 = update_lagrange(T, T_new, local_prices, rho)

        error = max(np.max(S), np.max(R))
        errors.append(error)

        #print(f"Maximum error at iteration {k}: {error}")
        #print("***************************************************")

        #end_time = time.time()
        #times.append(end_time - start_time)

        T = T_new.copy()
    
    return T, local_prices