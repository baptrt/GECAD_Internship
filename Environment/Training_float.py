from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from RL_float import PeerToPeerMarketEnv
env = PeerToPeerMarketEnv()

# ------------------------------
# 1. Create and wrap the training environment
train_env = DummyVecEnv([lambda: PeerToPeerMarketEnv()])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.)

check_env(PeerToPeerMarketEnv())  # check only once on a raw instance

# ------------------------------
# 2. Evaluation callback environment (separate, non-training)
eval_env = DummyVecEnv([lambda: PeerToPeerMarketEnv()])
eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=True, clip_obs=10.)

# ------------------------------
# 3. EvalCallback
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=os.path.join(log_dir, "best_model"),
    log_path=log_dir,
    eval_freq=1000,
    n_eval_episodes=5,
    deterministic=True,
    render=False
)

# ------------------------------
# 4. Train the model
policy_kwargs = dict(net_arch=[128, 128])
model = SAC(
    "MlpPolicy",
    train_env,
    policy_kwargs=policy_kwargs,
    batch_size=128,
    ent_coef='auto_1',
    verbose=1
)

# --- 1) Callback definition to recup rewards ---
class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self) -> bool:
        self.rewards.append(self.locals['rewards'][0])
        return True

# --- 2) Callback creation ---
reward_callback = RewardLoggerCallback()

# --- 3) Training ---
total_steps = 100_000

model.learn(total_timesteps=total_steps, callback=reward_callback)

def moving_average(x, window_size=50):
    return np.convolve(x, np.ones(window_size)/window_size, mode='valid')

# --- 4) Plot ---
plt.figure(figsize=(10,5))
plt.plot(reward_callback.rewards, alpha=0.3, label="Gross reward")
plt.plot(moving_average(reward_callback.rewards, 50), label="Moving average (50)", linewidth=2)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Evolution of reward during training (smoothed)")
plt.legend()
plt.grid()
plt.show()

df = pd.DataFrame({
    "step": range(len(reward_callback.rewards)),
    "reward": reward_callback.rewards
})
df.to_csv("reward_history_{total_step}.csv", index=False)
print("Rewards saved in reward_history_{total_step}.csv")

# ------------------------------
# 5. Save the model and the VecNormalize wrapper
model.save(os.path.join(log_dir, "best_model", "sac_model"))
train_env.save(os.path.join(log_dir, "best_model", "vecnormalize.pkl"))

# ------------------------------
# 6. Reload and test (optional, just to check all works)
loaded_env = DummyVecEnv([lambda: PeerToPeerMarketEnv()])
loaded_env = VecNormalize.load(os.path.join(log_dir, "best_model", "vecnormalize.pkl"), loaded_env)

loaded_env.training = False
loaded_env.norm_reward = False

model = SAC.load(os.path.join(log_dir, "best_model", "sac_model"), env=loaded_env)

obs = loaded_env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = loaded_env.step(action)
    print("Reward:", reward)
  