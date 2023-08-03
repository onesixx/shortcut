
import inspect
#-------------------------------------------------------------------------------
import gymnasium
from stable_baselines3 import A2C

env = gymnasium.make('LunarLander-v2',render_mode="rgb_array")
env.reset()
# ------------------------------------------------------------------------------
# for step in range(200):
#     #env.render()
#     some_action = env.action_space.sample()
#     print(f'action::{some_action}')
#     observation, reward, terminated, truncated, info = env.step(some_action)
#     print(f'obs::{observation}')
#     print(f'reward::{reward}')
#     #print(f'obs::{obs}, reward::{reward}, done::{done}, info::{info}')
# ------------------------------------------------------------------------------
model = A2C('MlpPolicy', env, verbose=1)
# model.learn(total_timesteps=10000)
# episodes = 10

model.learn(total_timesteps=100000)
episodes = 5
for ep in range(episodes):
	obs = env.reset()
	done = False
	while not done:
		# pass observation to model to get predicted action
		action, _states = model.predict(obs)
		# pass action to env and get info back
		obs, rewards, done, info = env.step(action)
		# show the environment on the screen
		env.render()
env.close()