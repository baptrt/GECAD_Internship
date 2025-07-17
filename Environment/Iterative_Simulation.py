## Dynamic Simulation

from Agent import initialize_agents
from Modelisation import simulate_market

import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Market parameters

n_agents = 10
T = np.zeros((n_agents+1, n_agents+1)) # Matrix of power exchanges
local_prices = np.zeros((n_agents+1, n_agents+1)) # Matrix of prices
gamma_0 = np.zeros((n_agents+1, n_agents+1)) # Matrix of penalties for exchanges

# Input to act on the market evolution

gamma = np.zeros((n_agents+1, n_agents+1)) # Matrix of penalties for exchanges

# Simulation parameters

rho = 10.0
rhol = 1.0
max_iters = 1000

for i in range(n_agents):      
    for j in range(n_agents):
        if i != j:
            gamma[i, j] = random.uniform(1, 5) # Random penalty between 0.01 and 0.05
        else:
            gamma[i, j] = 0
            
gamma = 0.5 * (gamma + gamma.T)

grid_price = 2
local_prices[:,n_agents] = grid_price
local_prices[n_agents,:] = grid_price

# Initialize agents

agents, a, b, tmin, tmax, pmin, pmax, n_producer, n_consumer, n_prosumer = initialize_agents(n_agents)
print("Agents initialisés avec succès")

# Simulation of the market with local prices pertubation

step_time = 10
times = np.zeros(step_time)

Local_Powers = []
Local_Prices = []
Grid_Powers = []
Grid_Prices = []

for i in range(step_time):
    
    print("Step number", i)
    times[i] = i
    
    prioritary_agents = random.sample(range(n_agents), 3)
    print("Agents prioritaires :", prioritary_agents)

    for k in range(n_agents):      
        for j in range(n_agents):
            if k != j:
                base_penalty = random.uniform(1, 10) # Random penalty between 1 and 10
                if k in prioritary_agents or j in prioritary_agents:
                    gamma[k, j] = base_penalty * 0.2  
                else:
                    gamma[k, j] = base_penalty
            else:
                gamma[k, j] = 0

    gamma = 0.5 * (gamma + gamma.T)  
    
    T_pertubated, lambda_pertubated = simulate_market(T, agents, max_iters, a, b, tmin, tmax, pmin, pmax, gamma, local_prices, rho, rhol)
    
    local_prices_pertubated = lambda_pertubated + gamma
    
    Local_Powers.append(T_pertubated)
    Local_Prices.append(local_prices_pertubated)
    Grid_Powers.append(T_pertubated[:, n_agents])
    Grid_Prices.append(local_prices_pertubated[:, n_agents])
    

print("\n Simulation of the market with local prices pertubation ended successfully \n")

local_prices_pertubated = lambda_pertubated + gamma

Pl_pertubated = sum(T_pertubated[:, n_agents])

# Outpout 

Outpout_scalar = [Pl_pertubated, grid_price]
Outpout_vector = [T_pertubated, local_prices_pertubated]

# Display

print("Number of producers :", n_producer)
print("Number of consumers :", n_consumer)
print("Number of prosumers :", n_prosumer)

print("Total power exchanged with the grid after perturbation :", Pl_pertubated)

def plot_matrix_with_colorbar(ax, matrix, cmap, title, label):
    im = ax.imshow(matrix, cmap=cmap, interpolation='nearest')
    ax.plot(range(n_agents), range(n_agents), color='black', linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Agent j")
    ax.set_ylabel("Agent i")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label=label)

# Exchange matrices display

fig, axs = plt.subplots(2, 5, figsize=(25, 10))  # 2 lignes x 5 colonnes
fig.suptitle("Evolution of the power exchanged between agents at each step", fontsize=16)

for i in range(step_time):
    row = i // 5
    col = i % 5
    ax = axs[row, col]
    im = ax.imshow(Local_Powers[i], cmap='RdBu', interpolation='nearest')
    ax.plot(range(n_agents), range(n_agents), color='black', linewidth=1)
    ax.set_title(f"Step {i}")
    ax.set_xlabel("Agent j")
    ax.set_ylabel("Agent i")
    
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)

plt.tight_layout(rect=[0, 0, 1, 0.95])  
plt.show()

# Price matrices display

fig, axs = plt.subplots(2, 5, figsize=(25, 10))  # 2 lignes x 5 colonnes
fig.suptitle("Evolution of the local prices at each step", fontsize=16)

for i in range(step_time):
    row = i // 5
    col = i % 5
    ax = axs[row, col]
    im = ax.imshow(Local_Prices[i], cmap='viridis', interpolation='nearest')
    ax.plot(range(n_agents), range(n_agents), color='black', linewidth=1)
    ax.set_title(f"Step {i}")
    ax.set_xlabel("Agent j")
    ax.set_ylabel("Agent i")
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

print("Simulation ended successfully!")