## ADMM implementation

import numpy as np

# Function to update the local variables of each agent

def admm_update(T, agents, max_iters,
                a, b, tmin, tmax, pmin, pmax,
                gamma, rho, rhol, bt1,
                max_error, P, Mu, T_mean):
    
    alpha = 0.7
    
    n, m = T.shape
    T_new = T.copy()
    
    n_consumer = 0
    n_producer = 0
    n_prosumer = 0
    for i in range(n):
        if agents[i].role == "producer":
            n_producer += 1
        elif agents[i].role == "consumer":
            n_consumer += 1
        else:
            n_prosumer += 1   
            
    for i in range(n):
        iter = 0
        erreur = 2 * max_error
        t_mean = p = mu = 0.0 # Initialization of the variables

        while (erreur > max_error) or iter < 2:
            iter += 1
            
            s = 0.0
            n_neighbors = len(agents[i].neighbors)
            list_err = np.zeros(n_neighbors)

            for k, neighbor in enumerate(agents[i].neighbors): 
                neighbor_id = neighbor.id

                bt2 = T[i, neighbor_id] - t_mean + p - mu
                T_new[i, neighbor_id] = (rho * bt1[i, neighbor_id] + bt2 * rhol - gamma[i, neighbor_id]) / (rho + rhol)
                T_new[i, neighbor_id] = max(min(T_new[i, neighbor_id], tmax[i]), tmin[i])
                s += T_new[i, neighbor_id]
                
                list_err[k] = abs(T_new[i, neighbor_id] - T[i, neighbor_id])
            
            erreur = np.max(list_err)
            t_mean = s / n_neighbors
            bp1 = mu + t_mean
            denom = n_neighbors * rhol + (n_neighbors**2) * a[i]
            p = (n_neighbors * rhol * bp1 - n_neighbors * b[i]) / denom
            p = max(min(p, pmax[i] / n_neighbors), pmin[i] / n_neighbors)
            mu = mu + alpha*(t_mean - p)
            erreur = max(erreur, abs(t_mean - p))
            
            T = T_new.copy()

        P[i] = p
        Mu[i] = mu
        T_mean[i] = t_mean

    return T, P, Mu, T_mean

# Function to update global variables

def update_lagrange(T, T_new, lambd, rho):
    n, m = T.shape
    R = np.zeros(n)
    S = np.zeros(n)
    bt1 = np.zeros((n, m))

    for i in range(n):
        r = s = 0
        for j in range(m):
            # Mise à jour classique pour les autres agents
            lambd[i, j] += rho / 2 * (T_new[i, j] + T_new[j, i])
            bt1[i, j] = 0.5 * (T_new[i, j] - T_new[j, i]) - lambd[i, j] / rho
            r += abs(T_new[i, j] + T_new[j, i])
            s += abs(T_new[i, j] - T[i, j])

        R[i] = r
        S[i] = s

    return R, S, lambd, bt1