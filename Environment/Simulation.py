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
            
# Simulation of the market without local prices pertubation

T_0, lambda_0 = simulate_market(T, agents, max_iters, a, b, tmin, tmax, pmin, pmax, gamma_0, local_prices, rho, rhol)

print("\n Simulation of the market without local prices pertubation ended successfully \n")

# Simulation of the market with local prices pertubation

T_pertubated, lambda_pertubated = simulate_market(T, agents, max_iters, a, b, tmin, tmax, pmin, pmax, gamma, local_prices, rho, rhol)

print("\n Simulation of the market with local prices pertubation ended successfully \n")

local_prices_0 = lambda_0 + gamma_0
local_prices_pertubated = lambda_pertubated + gamma

Pl_0 = sum(T_0[:, n_agents])

Pl_pertubated = sum(T_pertubated[:, n_agents])

# Outpout 

Outpout_scalar = [Pl_pertubated, grid_price]
Outpout_vector = [T_pertubated, local_prices_pertubated]

# Display

print("Number of producers :", n_producer)
print("Number of consumers :", n_consumer)
print("Number of prosumers :", n_prosumer)

print("Total power exchanged with the grid :", Pl_0)
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

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
plot_matrix_with_colorbar(axs[0], T_0, cmap='RdBu', title="T - Original", label="Power exchanged")
plot_matrix_with_colorbar(axs[1], T_pertubated, cmap='RdBu', title="T - Disturbed", label="Power exchanged")
plt.suptitle("Exhange Matrices - Original vs Disturbed")
plt.tight_layout()
plt.show()

# Price matrices display

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
plot_matrix_with_colorbar(axs[0], local_prices_0, cmap='viridis', title="Price - Original", label="Local Price")
plot_matrix_with_colorbar(axs[1], local_prices_pertubated, cmap='viridis', title="Price - Disturbed", label="Local Price")
plt.suptitle("Local Prices Matrices - Original vs Disturbed")
plt.tight_layout()
plt.show()

print("Simulation ended successfully!")