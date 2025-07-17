## Agent definition 

import numpy as np
import matplotlib.pyplot as plt
import random

# Define the Agent class

class Agent:
    def __init__(self, id, role):
        self.id = id
        self.role = role
        self.neighbors = []    
        #self.T = {}  # To store power exchanges of Multi-Agents Exchanges   
        
    def __repr__(self):
        return f"Agent(id={self.id}, role='{self.role}', neighbors={[n.id for n in self.neighbors]})"
        
    def add_neighbor(self, other_agent):
        # Vérifie que le voisin n'est pas déjà présent
        if other_agent.id not in [n.id for n in self.neighbors]:
            self.neighbors.append(other_agent)

        # Ajoute l'agent courant comme voisin de l'autre, si nécessaire
        if self.id not in [n.id for n in other_agent.neighbors]:
            other_agent.neighbors.append(self)
        
def cost_function(agent, a, b, c, P):
    ai = a[agent.id]
    bi = b[agent.id]
    ci = c[agent.id]
    return ai * P**2 + bi * P + ci

# Function to initialize agents
        
def initialize_agents(n_agents):
    agents = []
    roles = ["producer", "consumer", "prosumer"]
    
    for i in range(n_agents):
        role = random.choice(roles)
        agent = Agent(i, role)
        agents.append(agent)
        
    grid_agent = Agent(id=n_agents, role="grid")
    agents.append(grid_agent)
    n_agents += 1  
        
    for i in range(n_agents-1):
        agent = agents[i]
        for j in range(n_agents):
            if i != j:
                other = agents[j]
                if agent.role == "producer" and other.role != "producer":
                    agent.add_neighbor(other)
                elif agent.role == "consumer" and other.role != "consumer":
                    agent.add_neighbor(other)
                else:
                    agent.add_neighbor(other)
                    
    for i in range(n_agents-1):  # excludes grid_agent itself
        if i != n_agents :
            agents[i].add_neighbor(grid_agent)
            grid_agent.add_neighbor(agents[i]) # Add a neighbour to the grid_agent
    
    a = np.zeros(n_agents)
    b = np.zeros(n_agents)
    c = np.zeros(n_agents)
    tmax = np.zeros(n_agents)
    tmin = np.zeros(n_agents)
    pmax = np.zeros(n_agents)
    pmin = np.zeros(n_agents)
    n_producer, n_consumer, n_prosumer = 0, 0, 0
    
    for i in range(n_agents-1):
        if agents[i].role == "producer":
            a[i] = 0.01
            b[i] = -1
            c[i] = -0.01*15**2 + 15
            pmax[i] = 15
            pmin[i] = 1
            tmax[i] = pmax[i]
            tmin[i] = 0
            n_producer += 1
        elif agents[i].role == "consumer":
            a[i] = 0.01
            b[i] = 1
            c[i] = -0.01*15**2 + 15
            pmax[i] = -1
            pmin[i] = -15
            tmax[i] = 0
            tmin[i] = pmin[i]
            n_consumer += 1
        else:   # prosumer
            a[i] = 0.1
            b[i] = random.choice([-1, 1])
            c[i] = 2.5
            pmax[i] = 15
            pmin[i] = -15
            tmax[i] = pmax[i]
            tmin[i] = pmin[i]
            n_prosumer += 1
            
        a[grid_agent.id] = 0.2
        b[grid_agent.id] = 0
        pmin[grid_agent.id] = sum(pmin[:n_agents]) # can absorb any overproduction
        pmax[grid_agent.id] = sum(pmax[:n_agents]) # can provide any request
        tmin[grid_agent.id] = sum(tmin[:n_agents]) # no constraint on the minimum
        tmax[grid_agent.id] = sum(tmax[:n_agents]) # no constraint on the maximum

    return agents, a, b, tmin, tmax, pmin, pmax, n_producer, n_consumer, n_prosumer

def update_agents(agents):
    n_agents = len(agents)
    a = np.zeros(n_agents)
    b = np.zeros(n_agents)
    c = np.zeros(n_agents)
    tmax = np.zeros(n_agents)
    tmin = np.zeros(n_agents)
    pmax = np.zeros(n_agents)
    pmin = np.zeros(n_agents)
    for i in range(n_agents-1):
        if agents[i].role == "producer":
            a[i] = random.uniform(0, 0.1)  # Randomize the coefficient a
            b[i] = - random.uniform(0, 5)  # Randomize the coefficient b
            pmax[i] = 15
            pmin[i] = 0
            tmax[i] = pmax[i]
            tmin[i] = 0
            c[i] = - a[i]*pmax[i]**2 - b[i]*pmax[i]
        elif agents[i].role == "consumer":
            a[i] = random.uniform(0, 0.1)
            b[i] = random.uniform(0, 5)
            pmax[i] = 0
            pmin[i] = -15
            tmax[i] = 0
            tmin[i] = pmin[i]
            c[i] = - a[i]*pmin[i]**2 - b[i]*pmin[i]
        else:   # prosumer
            a[i] = 0.1
            b[i] = random.choice([-1, 1])
            c[i] = 2.5
            pmax[i] = 15
            pmin[i] = -15
            tmax[i] = pmax[i]
            tmin[i] = pmin[i]

    return agents, a, b, tmin, tmax, pmin, pmax

def initialize_test_2():
    agents = []    
    n_agents = 3
    
    roles = ["producteur", "consommateur", "grid"]
    for i in range(n_agents):
        role = roles[i]
        agent = Agent(i, role)
        agents.append(agent)

    a = np.zeros(n_agents)
    b = np.zeros(n_agents)
    tmax = np.zeros(n_agents)
    tmin = np.zeros(n_agents)
    pmax = np.zeros(n_agents)
    pmin = np.zeros(n_agents)

    for i in range(n_agents):
        if agents[i].role == roles[0]:
            a[i] = 1
            b[i] = -10
            pmax[i] = 60
            pmin[i] = 0
            tmax[i] = pmax[i]
            tmin[i] = pmin[i]
        elif agents[i].role == roles[1]:
            a[i] = 1
            b[i] = 20
            pmax[i] = 0
            pmin[i] = -30
            tmax[i] = pmax[i]
            tmin[i] = pmin[i]
        else:
            a[i] = 1
            b[i] = 10
            pmax[i] = np.inf
            pmin[i] = -np.inf
            tmax[i] = pmax[i]
            tmin[i] = pmin[i]
            
        for i in range(n_agents):
            agent = agents[i]
            for j in range(n_agents):
                if i != j:
                    other = agents[j]
                    if agent.role == "producer" and other.role != "producer":
                        agent.add_neighbor(other)
                    elif agent.role == "consumer" and other.role != "consumer":
                        agent.add_neighbor(other)
                    else:
                        agent.add_neighbor(other)     

    return agents, a, b, tmin, tmax, pmin, pmax 

    