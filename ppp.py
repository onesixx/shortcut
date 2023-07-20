# https://www.youtube.com/watch?v=XbWhJdQgi7E

import gym
#from customEnv import customEnv
from stable_baselines3 import A2C

env= gym.make("LunarLander-v2")
#env = customEnv()
env.reset()

model= A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000)

episodes =10

for ep in range(episodes):
    print(f'episode {ep}')
    obs = env.reset()     # instead step
    done = False
    while not done:
        env.render()
        a= env.action_space.sample()
        obs, reward, done, info, etc = env.step(a)
        print(f'action: {a} -- reward : {reward}--{etc}')

# 1 episode ------------------------------------------------
# for step in range(200):
#     env.render()
#     a= env.action_space.sample()
#     obs, reward, done, info, etc = env.step(a)
#     print(f'action: {a} -- reward : {reward}--{etc}')


# ---------------------------------------------------------
# print("action space ::", env.action_space)
# print("sample action ::", env.action_space.sample())

# print("obv space ::", env.observation_space)
# print("obv space shape ::", env.observation_space.shape)
# print("obv sample ::", env.observation_space.sample())

env.close()