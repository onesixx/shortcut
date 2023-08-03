
import inspect
import os

models_dir = "p_models/PPO"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

logdir = "p_logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)
#-------------------------------------------------------------------------------
import gymnasium
from stable_baselines3 import PPO

env = gymnasium.make('LunarLander-v2',render_mode="rgb_array")
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
# model.learn(total_timesteps=10000)
# episodes = 10

TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")

# TIMESTEPS = 10000
# iters = 0
# for i in range(30):
#     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
#     model.save(f"{models_dir}/{TIMESTEPS*i}")