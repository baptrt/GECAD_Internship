from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import os

import matplotlib.pyplot as plt
import numpy as np

from RL import PeerToPeerMarketEnv
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
best_model_dir = os.path.join(log_dir, "best_model")
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

# --- 1) Callback definition to recup rewards ---
class RewardLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards = []

    def _on_step(self) -> bool:
        self.rewards.append(self.locals['rewards'][0])
        return True

reward_callback = RewardLoggerCallback()

# ------------------------------
# 4. Load existing model if available
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

# --- 5) Training (continue training if model was loaded) ---
model.learn(total_timesteps=500_000, callback=[reward_callback, eval_callback])

def moving_average(x, window_size=50):
    return np.convolve(x, np.ones(window_size)/window_size, mode='valid')

# --- 6) Plot ---
plt.figure(figsize=(10,5))
plt.plot(reward_callback.rewards, alpha=0.3, label="Reward brut")
plt.plot(moving_average(reward_callback.rewards, 50), label="Moyenne glissante (50)", linewidth=2)
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("Évolution du reward (lissé)")
plt.legend()
plt.grid()
plt.show()

# ------------------------------
# 7. Save the model and the VecNormalize wrapper
model.save(model_path)
train_env.save(vecnorm_path)

# ------------------------------
# 8. Reload and test (optional)
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
