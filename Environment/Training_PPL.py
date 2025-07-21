import numpy as np
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from RL import PeerToPeerMarketEnv

# ------------------------------
# 1. Hyperparamètres initiaux
def sample_hyperparams():
    return {
        "learning_rate": float(np.random.uniform(1e-5, 1e-3)),
        "gamma": float(np.random.choice([0.95, 0.98, 0.99])),
        "ent_coef": float(np.random.uniform(0.001, 0.1))
    }

# ------------------------------
# 2. Entraîner un agent SAC
def train_sac_agent(env_id, params, timesteps=5000):
    env = DummyVecEnv([lambda: PeerToPeerMarketEnv()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=params["learning_rate"],
        gamma=params["gamma"],
        ent_coef=params["ent_coef"],
        verbose=0
    )
    model.learn(total_timesteps=timesteps)
    return model, env

# ------------------------------
# 3. Évaluer un agent
def evaluate(model, env, n_eval_episodes=5):
    rewards = []
    env.training = False
    env.norm_reward = False
    for _ in range(n_eval_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
        rewards.append(total_reward)
    env.training = True
    env.norm_reward = True
    return np.mean(rewards)

# ------------------------------
# 4. Population-Based Learning loop
N = 4  # Nombre d'agents dans la population
K = 3  # Nombre de cycles de sélection/mutation
T = 5000  # Nombre de timesteps par cycle

population = []
envs = []

# Initialisation
for i in range(N):
    params = sample_hyperparams()
    model, env = train_sac_agent(i, params, timesteps=T)
    population.append({"model": model, "params": params})
    envs.append(env)

# Cycles PPL
for cycle in range(K):
    print(f"\n=== Cycle {cycle + 1}/{K} ===")
    
    # Évaluation
    scores = [evaluate(agent["model"], envs[i]) for i, agent in enumerate(population)]
    sorted_indices = np.argsort(scores)[::-1]
    print("Scores:", scores)

    # Sélection du top 50%
    top_half = sorted_indices[:N // 2]

    # Remplacement des moins bons
    for i in sorted_indices[N // 2:]:
        j = np.random.choice(top_half)
        new_params = population[j]["params"].copy()
        # Mutation légère
        new_params["learning_rate"] *= np.random.uniform(0.8, 1.2)
        new_params["ent_coef"] *= np.random.uniform(0.8, 1.2)
        new_params["gamma"] = float(np.random.choice([0.95, 0.98, 0.99]))

        print(f"Remplacement de l'agent {i} par une mutation de l'agent {j}")
        model, env = train_sac_agent(i, new_params, timesteps=T)
        population[i] = {"model": model, "params": new_params}
        envs[i] = env

# ------------------------------
# 5. Sauvegarde du meilleur agent
final_scores = [evaluate(agent["model"], envs[i]) for i, agent in enumerate(population)]
best_index = np.argmax(final_scores)
best_agent = population[best_index]

print(f"\n🏆 Meilleur agent : {best_index}, score = {final_scores[best_index]:.2f}")
os.makedirs("best_agent", exist_ok=True)
best_agent["model"].save("best_agent_PPL/sac_model.zip")
envs[best_index].save("best_agent_PPL/vecnormalize.pkl")
