import gymnasium as gym
from gymnasium import spaces
import numpy as np

from Agent import initialize_test_2
from Modelisation_step import simulate_market_step

class PeerToPeerMarketEnv(gym.Env):
    def __init__(self, n_agents=2, verbose=True):
        super().__init__()

        self.n_agents = n_agents
        self.gamma_dim = (n_agents + 1, n_agents + 1)  # Including the grid agent
        self.max_gamma = 100.0
        self.min_gamma = -100.0
        self.current_step = 0
        self.final_reward = 0.0
        self.max_steps = 10
        self.max_error = 1e-3  # Maximum error threshold for termination
        self.last_gamma = np.zeros(self.gamma_dim)  # Last gamma matrix
        self.verbose = verbose  # Pour activer ou désactiver les prints

        # Observation: vector including prices and power exchanges
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=((self.n_agents + 1) * 4 + (self.n_agents + 1)**2,),
            dtype=np.float32
        )

        # Action: the gamma matrix (price signal)
        self.action_space = spaces.Box(
            low=self.min_gamma, high=self.max_gamma, shape=self.gamma_dim, dtype=np.float32
        )

        self.reset()
        
    # def step(self, action):
    #     done = False
    #     step = 0
    #     total_reward = 0
    #     gamma = self.last_gamma.copy()

    #     while self.error > self.max_error and step < self.max_iters:
    #         obs = self._get_obs()

    #         # Action RL pour cette itération
    #         gamma = 0.5 * (action + action.T)
    #         np.fill_diagonal(gamma, 0.0)
    #         self.last_gamma = gamma.copy()

    #         # Simulation avec gamma courant
    #         self.T, self.local_prices, self.bt1, self.P, self.Mu, self.T_mean, self.error = simulate_market_step(
    #             self.T, self.agents, self.max_error, self.a, self.b, self.tmin, self.tmax,
    #             self.pmin, self.pmax, gamma, self.local_prices,
    #             self.rho, self.rhol,
    #             self.bt1, self.P, self.Mu, self.T_mean)

    #         # Récompense pour cette itération
    #         reward = self._compute_reward(self.T)
    #         total_reward += reward

    #         if self.verbose:
    #             print(f"\n--- Étape {step} ---")
    #             print(f"Gamma:\n{np.round(gamma, 2)}")
    #             print(f"P_l = {np.sum(np.abs(self.T[:, -1])):.4f}, reward = {reward:.4f}, error = {self.error:.4e}")

    #         step += 1

    #     obs = self._get_obs()
    #     done = True  # car la boucle de convergence est terminée
    #     info = {"iterations": step}

    #     return obs, total_reward, done, False, info
    
    def reset_market_state(self):
        # Resets all market variables to their initial state
        T = np.zeros((self.n_agents + 1, self.n_agents + 1))  
        local_prices = np.zeros((self.n_agents + 1, self.n_agents + 1))
        bt1 = np.zeros((self.n_agents + 1, self.n_agents + 1))
        P = np.zeros(self.n_agents + 1)
        Mu = np.zeros(self.n_agents + 1)
        T_mean = np.zeros(self.n_agents + 1)
        return T, local_prices, bt1, P, Mu, T_mean

    
    def step(self, action):
        self.current_step += 1

        # Apply symmetrisation
        gamma = 0.5 * (action + action.T)
        np.fill_diagonal(gamma, 0.0)
        self.last_gamma = gamma.copy()

        # Initialise the state of the market
        T, local_prices, bt1, P, Mu, T_mean = self.reset_market_state()
        error = float('inf')
        internal_step = 0

        while error > self.max_error and internal_step < self.max_iters:
            T, local_prices, bt1, P, Mu, T_mean, error = simulate_market_step(
                T, self.agents, self.max_error, self.a, self.b, self.tmin, self.tmax,
                self.pmin, self.pmax, gamma, local_prices, self.rho, self.rhol,
                bt1, P, Mu, T_mean
            )
            internal_step += 1

        # Calculation of reward
        reward = self._compute_reward(T)
        
        # We store the reward but only give it out at the end
        self.final_reward = reward if self.current_step == self.max_steps else self.final_reward

        if self.verbose:
            print(f"\n--- Step {self.current_step} ---")
            print(f"Gamma:\n{np.round(gamma, 2)}")
            print(f"Reward: {reward:.4f}, P_l = {np.sum(np.abs(T[:, -1])):.4f}")

        obs = self._get_obs()
        terminated = self.current_step >= self.max_steps
        truncated = internal_step >= self.max_iters

        # Delayed reward
        return obs, self.final_reward if terminated else 0.0, terminated, truncated, {
            "internal_steps": internal_step,
            "final_error": error,
            "current_step": self.current_step
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.final_reward = 0.0

        self.T = np.zeros((self.n_agents + 1, self.n_agents + 1))
        self.local_prices = np.zeros_like(self.T)
        self.local_prices[:, -1] = 2
        self.local_prices[-1, :] = 2
        
        self.bt1 = np.zeros_like(self.T)
        self.P = np.zeros(self.n_agents + 1)
        self.Mu = np.zeros(self.n_agents + 1)
        self.T_mean = np.zeros(self.n_agents + 1)
        self.last_gamma = np.zeros((self.n_agents + 1, self.n_agents + 1))
        self.error = 2 * self.max_error


        self.rho = 10.0
        self.rhol = 1.0
        self.max_iters = 10000

        self.agents, self.a, self.b, self.tmin, self.tmax, self.pmin, self.pmax, *_ = initialize_test_2()

        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        # Vecteur de puissance vers le grid
        power_grid = self.T[:, -1]                          # (n+1,)
        # Prix appliqué par le grid
        price_grid = self.local_prices[:, -1]               # (n+1,)
        # Somme des échanges et des prix
        trade_total  = np.sum(self.T, axis=1)               # (n+1,)
        price_total  = np.sum(self.local_prices, axis=1)    # (n+1,)
        # Gamma précédent, normalisé
        gamma_flat   = (self.last_gamma / self.max_gamma).flatten()  # ((n+1)**2,)

        obs = np.concatenate([
            gamma_flat,
            power_grid,
            price_grid,
            trade_total,
            price_total
        ]).astype(np.float32)

        return obs

    # def _compute_reward(self, T):
    #     P_l = np.sum(np.abs(T[:, -1]))  # power exchanged with the network
    #     P_l_bar = 9
    #     alpha = 0.05
    #     beta = 0.9

    #     Penalisation sur la norme de gamma
    #     gamma_norm = np.linalg.norm(self.last_gamma) / (self.n_agents + 1)**2  # normalisée

    #     if P_l < P_l_bar:
    #         return 10 * beta * P_l 
    #     else:
    #         return - 100 * beta * (P_l - P_l_bar) 
    def _compute_reward(self, T):
        # Power exchanged with the grid
        P_l = np.sum(np.abs(T[:, -1]))
        P_l_bar = 9.0

        # Coefficients de pondération
        alpha = 0.0  # penalisation sur norme de gamma
        beta = 1    # poids sur l’énergie totale
        delta = 0.05  # pénalité sur variation de gamma

        # Norme de gamma
        gamma_norm = np.linalg.norm(self.last_gamma) / ((self.n_agents + 1) ** 2)

        # Variation de gamma par rapport au step précédent
        if hasattr(self, "prev_gamma"):
            gamma_delta = np.linalg.norm(self.last_gamma - self.prev_gamma) / ((self.n_agents + 1) ** 2)
        else:
            gamma_delta = 0.0

        # Mise à jour de prev_gamma
        self.prev_gamma = self.last_gamma.copy()

        # Composantes de récompense
        margin = P_l - P_l_bar
        if margin <= 0:
            reward_energy = + beta * (P_l**2) / P_l_bar**2
        else:
            reward_energy = - beta * (margin**2) / P_l_bar**2
        
        reward = reward_energy - delta * gamma_delta

        return reward
