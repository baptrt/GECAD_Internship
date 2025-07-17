## Modelisation of the market at one time step

from ADMM_step import admm_update, update_lagrange

import numpy as np
import matplotlib.pyplot as plt
# import time
# from tabulate import tabulate

def simulate_market_step(T, agents, max_iters,a, b, tmin, tmax, pmin, pmax, gamma, local_prices, rho, rhol, bt1, P, Mu, T_mean, max_error=1e-8):
    # Local update
    T_new, P, Mu, T_mean = admm_update(T, agents, max_iters, a, b, tmin, tmax, pmin, pmax, gamma, rho, rhol, bt1, max_error, P, Mu, T_mean)

    # Global update
    R, S, local_prices, bt1 = update_lagrange(T, T_new, local_prices, rho)

    error = max(np.max(S), np.max(R))
    T = T_new.copy()

    return T, local_prices, bt1, P, Mu, T_mean, error
