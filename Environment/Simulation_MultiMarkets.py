## Dynamic Simulation

from Agent import initialize_test_2
from Modelisation_step import simulate_market_step
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv

from RL import PeerToPeerMarketEnv  
from stable_baselines3 import SAC

# Loads the standardised driven env
env_ = PeerToPeerMarketEnv()
env = DummyVecEnv([lambda: PeerToPeerMarketEnv()])
env = VecNormalize.load("logs/best_model_Gaussian_1M/vecnormalize.pkl", env)
# env = VecNormalize.load("logs/best_model/vecnormalize.pkl", env)

# Very important: ensure that the normalisation stats never change again
env.training = False
env.norm_reward = False
obs = env.reset()
model = SAC.load("logs/best_model_Gaussian_1M/sac_model")
# model = SAC.load("logs/best_model/best_model")

# Market parameters

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
max_error = 1e-3
max_iters = 100000
max_gamma = 200.0

num_simulations = 100
Pl_final_values = []
Pl_final_configurations = []
Pl_history_all = []

Pl_history_all_0 = []
Pl_final_values_0 = []
Pl_final_configurations_0 = []

Pl = 0
Pl_0 = 0
for sim in range(num_simulations):
    print(f"\n===== Simulation {sim + 1} / {num_simulations} =====")

    agents, a, b, tmin, tmax, pmin, pmax = initialize_test_2()

    T = np.zeros((n_agents + 1, n_agents + 1))
    local_prices = np.zeros((n_agents + 1, n_agents + 1))
    gamma = np.zeros((n_agents + 1, n_agents + 1))

    bt1 = np.zeros_like(T)
    P = np.zeros(n_agents + 1)
    Mu = np.zeros(n_agents + 1)
    T_mean = np.zeros(n_agents + 1)
    error = 2 * max_error
    step = 0

    Pl_history_rl = []

    while error > max_error and step < max_iters:
        gamma = update_gamma_with_rl(agents, T, local_prices, gamma, max_gamma, model)

        T, local_prices, bt1, P, Mu, T_mean, error = simulate_market_step(
            T, agents, max_error, a, b, tmin, tmax, pmin, pmax,
            gamma, local_prices, rho, rhol, bt1, P, Mu, T_mean
        )

        Pl = np.sum(np.abs(T[:, -1]))
        Pl_history_rl.append(Pl)

        step += 1

    # Fin d'une simulation : stockage des données
    Pl_history_all.append(Pl_history_rl)
    Pl_final_values.append(Pl)
    Pl_final_configurations.append({
        "agents": agents,
        "T": T.copy(),
        "gamma": gamma.copy(),
        "local_prices": local_prices.copy(),
        "Pl": Pl
    })

# Overall analysis
Pl_array = np.array(Pl_final_values)
Pl_mean = np.mean(Pl_array)
Pl_std = np.std(Pl_array)
Pl_min_idx = np.argmin(Pl_array)
Pl_max_idx = np.argmax(Pl_array)

congestion_threshold = 3.0
num_curves = len(Pl_history_all)

Pl_trimmed = [np.array(p[1:]) for p in Pl_history_all if len(p) > 1]

Pl_final_trimmed = [p[-1] for p in Pl_trimmed]
Pl_min_idx = np.argmin(Pl_final_trimmed)
Pl_max_idx = np.argmax(Pl_final_trimmed)

max_len = max(len(p) for p in Pl_trimmed)
Pl_aligned = np.array([np.pad(p, (0, max_len - len(p)), constant_values=np.nan) for p in Pl_trimmed])

Pl_mean = np.nanmean(Pl_aligned, axis=0)
Pl_min = Pl_trimmed[Pl_min_idx]
Pl_max = Pl_trimmed[Pl_max_idx]
Pl_convergence_mean = np.mean(Pl_final_trimmed)
iterations = list(range(1, max_len + 1))

# --- PLOT ---
plt.figure(figsize=(12, 6))

# Individual curves in transparency
for i, p in enumerate(Pl_trimmed):
    label = "Simulated Markets" if i == 0 else None
    plt.plot(range(1, len(p) + 1), p, color='gray', alpha=0.2, label=label)

# Extremes
plt.plot(range(1, len(Pl_min) + 1), Pl_min, color='blue', linewidth=2, label='Minimum Power Exchanged')
plt.plot(range(1, len(Pl_max) + 1), Pl_max, color='red', linewidth=2, label='Maximum Power Exchanged')

# Congestion threshold line
plt.axhline(congestion_threshold, color='red', linestyle='--', label='Congestion Threshold')

# Black dotted horizontal line for the convergence average
plt.axhline(Pl_convergence_mean, color='black', linestyle='--', linewidth=1.5, label='Mean Power Exchanged at Convergence')

# Arrows on the extremes and the average
plt.annotate(f"{Pl_min[-1]:.2f} a.u.",
             xy=(len(Pl_min), Pl_min[-1]),
             xytext=(len(Pl_min)-10, Pl_min[-1] + 1),
             arrowprops=dict(arrowstyle='->', color='blue'),
             color='blue')

plt.annotate(f"{Pl_max[-1]:.2f} a.u.",
             xy=(len(Pl_max), Pl_max[-1]),
             xytext=(len(Pl_max)-10, Pl_max[-1] + 1),
             arrowprops=dict(arrowstyle='->', color='red'),
             color='red')

# Average of final values (convergence)
Pl_convergence_mean = np.mean(Pl_final_trimmed)

# Annotation of the convergence average (displayed on the far right of the graph)
plt.annotate(f"{Pl_convergence_mean:.2f} a.u.",
             xy=(max_len, Pl_convergence_mean),
             xytext=(max_len - 10, Pl_convergence_mean + 1),
             arrowprops=dict(arrowstyle='->', color='black'),
             color='black')

plt.title("Evolution of Power Exchanged with the DSO on Multiple Markets")
plt.xlabel("Iterations")
plt.ylabel("Total power exchanged with the DSO (a.u.)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()