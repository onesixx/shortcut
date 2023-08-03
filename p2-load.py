# https://www.youtube.com/watch?v=XbWhJdQgi7E

import gym
from stable_baselines3 import PPO, A2C
import os

myAlgorithm = "PPO"

model_dir = f"models/{myAlgorithm}"
log_dir =   "logs"
model_path = f"{model_dir}/170000.zip"


env= gym.make("LunarLander-v2", render_mode="rgb_array")
env.reset()

## Training & Saving
# if myAlgorithm =='PPO' :
#     model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
# elif myAlgorithm == 'A2C':
#     model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

## Load model
model = PPO.load(model_path, env=env)

episodes =10
for ep in range(episodes):
    print(f'episode {ep}===========================================')
    obs = env.reset()     # instead step
    done = False
    while not done:
        env.render()

        # a= env.action_space.sample()
        a, _ = model.predict(obs)
        obs, reward, done, info, etc = env.step(a)
        print(f'action: {a} -- reward : {reward}--{etc}')


env.close()
