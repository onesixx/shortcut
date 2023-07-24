from stable_baselines3 import PPO
import os
from snake_env import SnekEnv
import time

models_dir=f"models/{int(time.time())}/"
log_dir = f"logs/{int(time.time())}/"

env = SnekEnv()
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)

TIMESTEPS = 10000
iters = 0
while True:
    iters +=1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")

# !tensorboard --logdir logs --port 6001