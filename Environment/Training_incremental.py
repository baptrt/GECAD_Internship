from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from RL import PeerToPeerMarketEnv

# ------------------------------
# 1. Create and wrap the training environment
train_env = DummyVecEnv([lambda: PeerToPeerMarketEnv()])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

# Check the raw env only once
check_env(PeerToPeerMarketEnv())

# ------------------------------
# 2. Evaluation callback environment
eval_env = DummyVecEnv([lambda: PeerToPeerMarketEnv()])
eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True, clip_obs=10.)

# Sync normalization stats
eval_env.obs_rms = train_env.obs_rms

# ------------------------------
# 3. EvalCallback
log_dir = "./logs"
best_model_dir = os.path.join(log_dir, "best_model_Gaussian_1M")
os.makedirs(best_model_dir, exist_ok=True)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=best_model_dir,
    log_path=log_dir,
    eval_freq=1000,
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

# --- 4) Callback to collect rewards ---
class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self) -> bool:
        self.rewards.append(self.locals['rewards'][0])
        return True

reward_callback = RewardLoggerCallback()

# ------------------------------
# 5. Load existing model if available
policy_kwargs = dict(net_arch=[128, 128])
model_path = os.path.join(best_model_dir, "sac_model")
vecnorm_path = os.path.join(best_model_dir, "vecnormalize.pkl")

if os.path.exists(model_path + ".zip") and os.path.exists(vecnorm_path):
    print("=== Loading existing model and VecNormalize ===")
    train_env = VecNormalize.load(vecnorm_path, train_env)
    model = SAC.load(model_path, env=train_env)
else:
    print("=== No existing model found, training from scratch ===")
    model = SAC(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        batch_size=128,
        ent_coef='auto_1',
        verbose=1
    )

# ------------------------------
# 6. Training
model.learn(total_timesteps=500_000, callback=[reward_callback, eval_callback])

# ------------------------------
# 7. Plot rewards
def moving_average(x, window_size=50):
    return np.convolve(x, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(10,5))
plt.plot(reward_callback.rewards, alpha=0.3, label="Gross reward")
plt.plot(moving_average(reward_callback.rewards, 50), label="Moving averaging (50)", linewidth=2)
plt.xlabel("Training Step")
plt.ylabel("Reward")
plt.title("Évolution of reward durinf training (smoothed)")
plt.legend()
plt.grid()
plt.show()

# ------------------------------
# 8. Save model and VecNormalize
model.save(model_path)
train_env.save(vecnorm_path)

rewards_df = pd.DataFrame({
    "step": np.arange(len(reward_callback.rewards)),
    "reward": reward_callback.rewards
})

rewards_csv_path = os.path.join(log_dir, "rewards.csv")
rewards_df.to_csv(rewards_csv_path, index=False)
print(f"Rewards saved to {rewards_csv_path}")

# ------------------------------
# 9. Reload and test (optional)
loaded_env = DummyVecEnv([lambda: PeerToPeerMarketEnv()])
loaded_env = VecNormalize.load(vecnorm_path, loaded_env)
loaded_env.training = False
loaded_env.norm_reward = False

model = SAC.load(model_path, env=loaded_env)

obs = loaded_env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = loaded_env.step(action)
    print("Reward:", reward)
