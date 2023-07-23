
import gymnasium

env = gymnasium.make("LunarLander-v2"
                     ,render_mode='rgb_array') #"human")
env.reset()

# 1 episode ------------------------------------------------
for step in range(200):
    env.render()
    a= env.action_space.sample()
    obs, reward, done, info, etc = env.step(a)
    print(f'action: {a} -- reward : {reward}--{etc}')


# ---------------------------------------------------------
# print("action space ::",  env.action_space)
# print("sample action ::", env.action_space.sample())

# print("obv space ::",       env.observation_space)
# print("obv space shape ::", env.observation_space.shape)
# print("obv sample ::",      env.observation_space.sample())

env.close()
