## Dynamic Simulation

from Agent import initialize_test_2
from Modelisation_step import simulate_market_step
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import numpy as np
import matplotlib.pyplot as plt
import csv

from RL import PeerToPeerMarketEnv  
from stable_baselines3 import SAC

# Loads the standardised driven env
env = DummyVecEnv([lambda: PeerToPeerMarketEnv()])
env = VecNormalize.load("logs/best_model/vecnormalize.pkl", env)
# env = VecNormalize.load("logs/best_model/vecnormalize.pkl", env)

# Very important: ensure that the normalisation stats never change again
env.training = False
env.norm_reward = False
obs = env.reset()
model = SAC.load("logs/best_model/sac_model")
# model = SAC.load("logs/best_model/best_model")

# Market parameters

n_runs = 100  # Nombre de simulations
max_length = 200  # Longueur max des itérations autorisées (pour normalisation des tailles)

all_Pl_with_signal = []
all_Pl_without_signal = []
max_error = 1e-3
max_gamma = 200.0

n_agents = 2
P_l_bar = 3

verbose = True

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
    
    # print("Observation:", obs)

    return obs

def update_gamma_with_rl(agents, T, local_prices, gamma, max_gamma, rl_model):
    obs = get_obs_from_matrix(T, local_prices, gamma, max_gamma)
    obs = env.normalize_obs(obs)

    action, _ = rl_model.predict(obs, deterministic=True)
    action = action.reshape(gamma.shape)

    gamma = 0.5 * (action + action.T)
    np.fill_diagonal(gamma, 0.0)

    return gamma

# Simulation parameters

rho = 10.0
rhol = 1.0
max_iters = 100000

agents, a, b, tmin, tmax, pmin, pmax = initialize_test_2()
print("Agents initialisés avec succès")
print("Agents:", agents)
            
# ----------- Simulation of the market with signal prices -----------
for run in range(n_runs):
    print(f"\n--- Simulation {run + 1}/{n_runs} ---")

    agents, a, b, tmin, tmax, pmin, pmax = initialize_test_2()

    T = np.zeros((n_agents+1, n_agents+1))
    local_prices = np.zeros((n_agents+1, n_agents+1))
    gamma = np.zeros((n_agents+1, n_agents+1))
    bt1 = np.zeros_like(T)
    P = np.zeros(n_agents + 1)
    Mu = np.zeros(n_agents + 1)
    T_mean = np.zeros(n_agents + 1)
    error = 2 * max_error
    step = 0

    Pl_history_rl = []

    while error > max_error and step < max_length:
        gamma = update_gamma_with_rl(agents, T, local_prices, gamma, max_gamma, model)

        T, local_prices, bt1, P, Mu, T_mean, error = simulate_market_step(
            T, agents, max_error, a, b, tmin, tmax, pmin, pmax,
            gamma, local_prices, rho, rhol, bt1, P, Mu, T_mean
        )

        Pl = np.sum(np.abs(T[:, -1]))
        Pl_history_rl.append(Pl)
        step += 1

    while len(Pl_history_rl) < max_length:
        Pl_history_rl.append(Pl_history_rl[-1])

    all_Pl_with_signal.append(Pl_history_rl)


    # ----------- Simulation of the market without signal prices -----------
    T_0 = np.zeros((n_agents+1, n_agents+1))
    local_prices_0 = np.zeros((n_agents+1, n_agents+1))
    gamma_0 = np.zeros((n_agents+1, n_agents+1))
    bt1_0 = np.zeros_like(T_0)
    P_0 = np.zeros(n_agents + 1)
    Mu_0 = np.zeros(n_agents + 1)
    T_mean_0 = np.zeros(n_agents + 1)
    error_0 = 2 * max_error
    step_0 = 0

    Pl_history = []

    while error_0 > max_error and step_0 < max_length:
        T_0, local_prices_0, bt1_0, P_0, Mu_0, T_mean_0, error_0 = simulate_market_step(
            T_0, agents, max_error, a, b, tmin, tmax, pmin, pmax,
            gamma_0, local_prices_0, rho, rhol, bt1_0, P_0, Mu_0, T_mean_0
        )

        Pl_0 = np.sum(np.abs(T_0[:, -1]))
        Pl_history.append(Pl_0)
        step_0 += 1

    while len(Pl_history) < max_length:
        Pl_history.append(Pl_history[-1])

    all_Pl_without_signal.append(Pl_history)

all_Pl_with_signal = np.array(all_Pl_with_signal)       # (n_runs, max_length)
all_Pl_without_signal = np.array(all_Pl_without_signal) # (n_runs, max_length)

mean_with_signal = np.mean(all_Pl_with_signal, axis=0)
min_with_signal = np.min(all_Pl_with_signal, axis=0)
max_with_signal = np.max(all_Pl_with_signal, axis=0)

mean_without_signal = np.mean(all_Pl_without_signal, axis=0)
min_without_signal = np.min(all_Pl_without_signal, axis=0)
max_without_signal = np.max(all_Pl_without_signal, axis=0)

iterations = np.arange(max_length)

plt.figure(figsize=(10, 6))

plt.plot(iterations, mean_with_signal, label="Mean (with price signal)", color="blue")
plt.fill_between(iterations, min_with_signal, max_with_signal, color="blue", alpha=0.2)

plt.plot(iterations, mean_without_signal, label="Mean (without price signal)", color="green")
plt.fill_between(iterations, min_without_signal, max_without_signal, color="green", alpha=0.2)

plt.axhline(P_l_bar, color="red", linestyle="--", label="Congestion Threshold")

last_idx = max_length - 1

plt.annotate(
    f"With signal:\nMean={mean_with_signal[last_idx]:.2f}\nMin={min_with_signal[last_idx]:.2f}\nMax={max_with_signal[last_idx]:.2f}",
    xy=(last_idx, mean_with_signal[last_idx]),
    xytext=(last_idx - 20, mean_with_signal[last_idx] + 1),
    fontsize=10,
    color="blue",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)

plt.annotate(
    f"Without signal:\nMean={mean_without_signal[last_idx]:.2f}\nMin={min_without_signal[last_idx]:.2f}\nMax={max_without_signal[last_idx]:.2f}",
    xy=(last_idx, mean_without_signal[last_idx]),
    xytext=(last_idx - 20, mean_without_signal[last_idx] - 2.5),
    fontsize=10,
    color="green",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)

plt.xlabel("Iterations")
plt.ylabel("Total power exchanged with DSO (a.u.)")
plt.title(f"Average of {n_runs} simulations")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
