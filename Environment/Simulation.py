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
env = VecNormalize.load("logs/best_model/vecnormalize.pkl", env)
# env = VecNormalize.load("logs/best_model/vecnormalize.pkl", env)

# Very important: ensure that the normalisation stats never change again
env.training = False
env.norm_reward = False
obs = env.reset()
model = SAC.load("logs/best_model/best_model")
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
reward_history = []
gamma_history = []  # Stocke gamma à chaque itération

# --- Dynamic loop with gamma update ---
while error > max_error and step < max_iters:
    gamma_history.append(gamma.copy())

    if verbose:
        print(f"\n--- Étape {step} ---")
        print("Gamma (sym):\n", np.round(gamma, 2))
        
    gamma = update_gamma_with_rl(agents, T, local_prices, gamma, max_gamma, model)
    reward = env_._compute_reward(T)
    reward_history.append(reward)

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
        
    # gamma = update_gamma_with_rl(agents, T, local_prices, gamma, max_gamma, model)

    # --- Simulation of a stage ---
    T_0, local_prices_0, bt1_0, P_0, Mu_0, T_mean_0, error_0 = simulate_market_step(
        T_0, agents, max_error, a, b, tmin, tmax, pmin, pmax,
        gamma_0, local_prices_0, rho, rhol, bt1_0, P_0, Mu_0, T_mean_0
    )
    
    T_0_history.append(T_0.copy())
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

# Find the first iteration where the reward becomes negative.
count = 0
for i, r in enumerate(reward_history):
    if r < 0:
        count += 1
        if count == 2:
            iter_reward_neg = i
            val_reward_neg = r
            break

plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(reward_history)), reward_history, color="purple")
plt.axvline(x=iter_reward_neg, color='red', linestyle='--', linewidth=1.2, label="Reward becomes negative")
plt.plot(iter_reward_neg, val_reward_neg, marker='*', color='red', markersize=12, label="Negative Reward Onset")
plt.annotate(f"Negative reward\nstarts here ({iter_reward_neg})",
             xy=(iter_reward_neg, val_reward_neg),
             xytext=(iter_reward_neg + 3, val_reward_neg - 1),
             arrowprops=dict(arrowstyle="->", color='red'),
             fontsize=10, color='red')

plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)

plt.xlabel("Iterations")
plt.ylabel("Reward")
plt.title("Evolution of the reward")
plt.legend()
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
# Find the first crossing of the threshold
count = 0
for idx, (it, val) in enumerate(zip(iterations_rl, Pl_history_rl)):
    if val > P_l_bar:
        count += 1
        if count == 2:
            iter_cross = it
            val_cross = val
            break

plt.figure(figsize=(10, 6))
plt.plot(iterations_rl, Pl_history_rl, label="Market with Price Signal", color="blue", marker='x', markersize=6, linestyle='-')
plt.plot(iterations_no_signal, Pl_history, label="Market without Price Signal", color="green", marker='x', markersize=6, linestyle='-')
plt.axhline(P_l_bar, color="red", linestyle="--", label="Congestion Treshold", linewidth=1.5)
plt.axvline(x=iter_cross, color='purple', linestyle='--', linewidth=1.2, label="First Threshold Crossing")
plt.plot(iter_cross, val_cross, marker='*', markersize=12, color='black', label="Crossing Point")

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

plt.annotate("Threshold crossed", 
             xy=(iter_cross, val_cross), 
             xytext=(iter_cross + 2, val_cross + 1),
             arrowprops=dict(arrowstyle="->", color='purple'),
             fontsize=10, color='purple')


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

gamma_history = np.array(gamma_history)  # (steps, n_agents+1, n_agents+1)

n = n_agents + 1
steps = gamma_history.shape[0]

fig = plt.figure(figsize=(9, 6))
fig.suptitle("Evolution of Price Signal on i ↔ j exchanges by iteration", fontsize=16)

gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.5)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(range(steps), gamma_history[:, 2, 0])
ax1.set_title("Price Signal 2 ↔ 0")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Value")
ax1.grid(True, which='major', linestyle='-', linewidth=0.5)
ax1.minorticks_on()
ax1.grid(True, which='minor', linestyle=':', linewidth=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(range(steps), gamma_history[:, 1, 0])
ax2.set_title("Price Signal 1 ↔ 0")
ax2.set_xlabel("Iterations")
ax2.set_ylabel("Value")
ax2.grid(True, which='major', linestyle='-', linewidth=0.5)
ax2.minorticks_on()
ax2.grid(True, which='minor', linestyle=':', linewidth=0.3)

gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1, :], wspace=0.5)
ax3 = fig.add_subplot(gs_bottom[0])
ax3.plot(range(steps), gamma_history[:, 2, 1])
ax3.set_title("Price Signal 2 ↔ 1")
ax3.set_xlabel("Iterations")
ax3.set_ylabel("Value")
ax3.grid(True, which='major', linestyle='-', linewidth=0.5)
ax3.minorticks_on()
ax3.grid(True, which='minor', linestyle=':', linewidth=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
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
        
# Save T_0 history to CSV
t0_final_csv = "T_0_final_evolution.csv"
T_0_history = np.array(T_0_history)  # (steps, n_agents+1, n_agents+1)

with open(t0_final_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    header = ["Iteration"] + [f"T_0_{i}_{j}" for i in range(n_agents + 1) for j in range(n_agents + 1)]
    writer.writerow(header)
    for iteration, T_mat in enumerate(T_0_history):
        writer.writerow([iteration] + list(T_mat.flatten()))

print("CSV files exported:")
print("  - Pl_history_with_signal.csv")
print("  - Pl_history_without_signal.csv")
print(f"  - {t0_final_csv}")    
