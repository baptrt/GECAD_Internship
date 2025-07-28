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
    
    print("Observation:", obs)

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
            
###################### Simulation of the market with signal prices ######################

T = np.zeros((n_agents+1, n_agents+1)) # Matrix of power exchanges
local_prices = np.zeros((n_agents+1, n_agents+1)) # Matrix of prices
gamma = np.zeros((n_agents+1, n_agents+1)) # Matrix of penalties for exchanges

max_error = 1e-3
step = 0

bt1 = np.zeros_like(T)
P = np.zeros(n_agents + 1)
Mu = np.zeros(n_agents + 1)
T_mean = np.zeros(n_agents + 1)
error = 2 * max_error
max_gamma = 500  # Maximum value for gamma normalization

T_history = []  # List for storing the T matrix at each iteration
Pl_history_rl = []
local_prices_history = []    

# --- Dynamic loop with gamma update ---
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
    
    T_history.append(T.copy())
    local_prices_history.append(local_prices.copy())

    # --- Calculation of an indicator: P_l (exchange with the network) ---
    Pl = np.sum(np.abs(T[:, -1]))

    Pl_history_rl.append(Pl)

    print(f"P_l = {Pl:.4f}, error = {error:.4e}")
    print("T:\n", T)
    print("Local Prices:\n", local_prices)
    
    step += 1
    
    
###################### Simulation of the market without signal prices ######################
T_0 = np.zeros((n_agents+1, n_agents+1)) # Matrix of power exchanges
local_prices_0 = np.zeros((n_agents+1, n_agents+1)) # Matrix of prices
gamma_0 = np.zeros((n_agents+1, n_agents+1)) # Matrix of penalties for exchanges

max_error = 1e-3
step_0 = 0

bt1_0 = np.zeros_like(T)
P_0 = np.zeros(n_agents + 1)
Mu_0 = np.zeros(n_agents + 1)
T_mean_0 = np.zeros(n_agents + 1)
error_0 = 2 * max_error

T_0_history = []  # List for storing the T matrix at each iteration
Pl_history = []  # List for storing the P_l values at each iteration
local_prices_0_history = []   # Wihtout signal

# --- Dynamic loop ---    
while error_0 > max_error and step_0 < max_iters:

    if verbose:
        print(f"\n--- Étape {step} ---")
        print("Gamma (sym):\n", np.round(gamma, 2))
        
    gamma = update_gamma_with_rl(agents, T, local_prices, gamma, max_gamma, model)

    # --- Simulation of a stage ---
    T_0, local_prices_0, bt1_0, P_0, Mu_0, T_mean_0, error_0 = simulate_market_step(
        T_0, agents, max_error, a, b, tmin, tmax, pmin, pmax,
        gamma_0, local_prices_0, rho, rhol, bt1_0, P_0, Mu_0, T_mean_0
    )
    
    T_0_history.append(T.copy())
    local_prices_0_history.append(local_prices_0.copy())

    # --- Calculation of an indicator: P_l (exchange with the network) ---
    Pl_0 = np.sum(np.abs(T_0[:, -1]))

    Pl_history.append(Pl_0)

    print(f"P_l = {Pl:.4f}, error = {error:.4e}")
    print("T:\n", T)
    print("Local Prices:\n", local_prices)
    
    step_0 += 1


# --- Final Results ---
print("\nSimulation ended.")
print("T final:\n", T)
print("Price without signal:\n", local_prices)
print("Final prices:\n", local_prices+gamma)
print("Price Signal:", gamma)

print("\n Simulation of the market without local prices pertubation ended successfully \n")

print("Simulation ended successfully!")

plt.rcParams.update({'font.size': 14}) # Size

T_history = np.array(T_history)  # Shape: (steps, n_agents+1, n_agents+1)
T_0_history = np.array(T_0_history)  # Shape: (steps, n_agents+1, n_agents+1)

iterations = np.arange(T_history.shape[0])
iterations_0 = np.arange(T_0_history.shape[0])

for i in range(n_agents):
    plt.plot(iterations, T_history[:, i, -1], label=f"Agent {i} → DSO")

plt.xlabel("Iterations")
plt.ylabel("Power exchanged with the DSO (a.u.)")
plt.title("Evolution of power exchanged per agent with the DSO (with price signal)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Convert history into array
Pl_history_rl = np.array(Pl_history_rl)
Pl_history = np.array(Pl_history)

# Create two distinct time axes
iterations_rl = np.arange(len(Pl_history_rl))
iterations_no_signal = np.arange(len(Pl_history))

# Latest values (convergence)
final_iter_rl = iterations_rl[-1]
final_iter_no_signal = iterations_no_signal[-1]
final_Pl_rl = Pl_history_rl[-1]
final_Pl_no_signal = Pl_history[-1]

# Draw the two curves 
plt.figure(figsize=(10, 6))
plt.plot(iterations_rl, Pl_history_rl, label="Market with Price Signal", color="blue", marker='x', markersize=6, linestyle='-')
plt.plot(iterations_no_signal, Pl_history, label="Market with Price Signal", color="green", marker='x', markersize=6, linestyle='-')
plt.axhline(P_l_bar, color="red", linestyle="--", label="Congestion Treshold", linewidth=1.5)

# Add annotations for final values
plt.annotate(f"{final_Pl_rl:.2f} a.u.", 
             xy=(final_iter_rl, final_Pl_rl), 
             xytext=(final_iter_rl - 5, final_Pl_rl + 1),
             arrowprops=dict(arrowstyle="->", color='blue'),
             fontsize=10, color='blue')

plt.annotate(f"{final_Pl_no_signal:.2f} a.u.", 
             xy=(final_iter_no_signal, final_Pl_no_signal), 
             xytext=(final_iter_no_signal - 5, final_Pl_no_signal + 1),
             arrowprops=dict(arrowstyle="->", color='green'),
             fontsize=10, color='green')

# Refined grid pattern
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)

plt.xlabel("Iterations")
plt.ylabel("Total power exchanged with the DSO (a.u.)")
plt.title("Comparison with and without a price signal")
plt.legend()
plt.tight_layout()
plt.show()

local_prices_history = np.array(local_prices_history)         
local_prices_0_history = np.array(local_prices_0_history)

plt.figure(figsize=(10, 6))
plt.plot(iterations, local_prices_history[:, 0, 1], label="Price Agent 0 ↔ Agent 1")
plt.plot(iterations, local_prices_history[:, 0, -1], label="Price Agent 0 ↔ DSO")
plt.plot(iterations, local_prices_history[:, 1, -1], label="Price Agent 1 ↔ DSO")

plt.xlabel("Iterations")
plt.ylabel("Local price")
plt.title("Price Evolution (with price signal)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(iterations_0, local_prices_0_history[:, 0, 1], label="Price Agent 0 ↔ Agent 1")
plt.plot(iterations_0, local_prices_0_history[:, 0, -1], label="Price Agent 0 ↔ DSO")
plt.plot(iterations_0, local_prices_0_history[:, 1, -1], label="Price Agent 1 ↔ DSO")

plt.xlabel("Iterations")
plt.ylabel("Local price")
plt.title("Price evolution (without price signal)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save P_l values WITH price signal (RL)
with open("Pl_history_with_signal.csv", mode="w", newline="") as file_rl:
    writer = csv.writer(file_rl)
    writer.writerow(["Iteration", "P_l_with_signal"])
    for i, pl in enumerate(Pl_history_rl):
        writer.writerow([i, pl])

# Save P_l values WITHOUT price signal
with open("Pl_history_without_signal.csv", mode="w", newline="") as file_no_signal:
    writer = csv.writer(file_no_signal)
    writer.writerow(["Iteration", "P_l_without_signal"])
    for i, pl in enumerate(Pl_history):
        writer.writerow([i, pl])

print("CSV files exported:")
print("  - Pl_history_with_signal.csv")
print("  - Pl_history_without_signal.csv")