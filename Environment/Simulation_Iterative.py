## Dynamic Simulation

from Agent import initialize_test_2
from Modelisation_step import simulate_market_step
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from RL import PeerToPeerMarketEnv  
from stable_baselines3 import SAC

# Charge l'env normalisé entraîné
env = DummyVecEnv([lambda: PeerToPeerMarketEnv()])
env = VecNormalize.load("logs/best_model/vecnormalize.pkl", env)

# Très important : assure que les stats de normalisation ne changent plus
env.training = False
env.norm_reward = False
obs = env.reset()
model = SAC.load("logs/best_model/best_model")


# Market parameters

n_agents = 2
T = np.zeros((n_agents+1, n_agents+1)) # Matrix of power exchanges
local_prices = np.zeros((n_agents+1, n_agents+1)) # Matrix of prices
gamma_0 = np.zeros((n_agents+1, n_agents+1)) # Matrix of penalties for exchanges

n_agents = 2
verbose = True
max_error = 1e-3
step = 0

bt1 = np.zeros_like(T)
P = np.zeros(n_agents + 1)
Mu = np.zeros(n_agents + 1)
T_mean = np.zeros(n_agents + 1)
error = 2 * max_error
max_gamma = 500  # Maximum value for gamma normalization
Pl_bar = 6
Pl = 10

def get_obs_from_matrix(T, local_prices, last_gamma, max_gamma):

    power_grid = T[:, -1]                      # (n+1,)
    price_grid = local_prices[:, -1]           # (n+1,)
    trade_total = np.sum(T, axis=1)             # (n+1,)
    price_total = np.sum(local_prices, axis=1)  # (n+1,)
    gamma_flat = (last_gamma / max_gamma).flatten()  # ((n+1)^2,)

    obs = np.concatenate([
        gamma_flat,
        power_grid,
        price_grid,
        trade_total,
        price_total
    ]).astype(np.float32)
    
    print("Observation:", obs)

    return obs

def update_gamma_with_rl(agents, T, local_prices, gamma, max_gamma, rl_model):
    n = len(agents)
    new_gamma = np.zeros_like(gamma)

    for i, agent in enumerate(agents):
        if agent.role == "grid":
            continue  

        obs = get_obs_from_matrix(T, local_prices, gamma, max_gamma)
        obs = env.normalize_obs(obs)

        output = rl_model.predict(obs, deterministic=True)
        action = output[0]
        print("Action RL:", action)

        action = np.array(action)

        for neighbor_agent in agent.neighbors:
            j = neighbor_agent.id  
            gamma_val = action[i, j]  
            new_gamma[i, j] = gamma_val

    # Make gamma symmetrical and diagonal-free
    gamma = 0.5 * (new_gamma + new_gamma.T)
    np.fill_diagonal(gamma, 0.0)

    return gamma

# Input to act on the market evolution

gamma = np.zeros((n_agents+1, n_agents+1)) # Matrix of penalties for exchanges

# Simulation parameters

rho = 10.0
rhol = 1.0
max_iters = 100000

# for i in range(n_agents):      
#     for j in range(n_agents):
#         if i != j:
#             gamma[i, j] = random.uniform(1, 5) # Random penalty between 0.01 and 0.05
#         else:
#             gamma[i, j] = 0
            
# gamma = 0.5 * (gamma + gamma.T)

# grid_price = 2
# local_prices[:,n_agents] = grid_price
# local_prices[n_agents,:] = grid_price

# Initialize agents

agents, a, b, tmin, tmax, pmin, pmax = initialize_test_2()
print("Agents successfully initialised")
print("Agents:", agents)
            
# Simulation of the market without local prices pertubation

#T_0, lambda_0 = simulate_market(T, agents, max_iters, a, b, tmin, tmax, pmin, pmax, gamma, local_prices, rho, rhol)

# --- Boucle dynamique avec mise à jour de gamma ---
while error > max_error and step < max_iters:

    if verbose:
        print(f"\n--- Étape {step} ---")
        print("Gamma (sym):\n", np.round(gamma, 2))
        
    gamma = update_gamma_with_rl(agents, T, local_prices, gamma, max_gamma, model)

    # --- Simulation of a stage ---
    T, local_prices, bt1, P, Mu, T_mean, error = simulate_market_step(
        T, agents, max_error, a, b, tmin, tmax, pmin, pmax,
        gamma, local_prices, rho, rhol, bt1, P, Mu, T_mean
    )

    # --- Calculation of an indicator: P_l (exchange with the network) ---
    Pl = np.sum(np.abs(T[:, -1]))

    print(f"P_l = {Pl:.4f}, error = {error:.4e}")
    print("T:\n", T)
    
    step += 1

# --- Final results---
print("\nSimulation complete.")
print("T final:\n", T)
print("Price without signal:\n", local_prices)
print("Final prices:\n", local_prices+gamma)
print("Price signal :", gamma)


print("\n Simulation of the market without local prices pertubation ended successfully \n")

print("Simulation ended successfully!")