# https://www.youtube.com/watch?v=XbWhJdQgi7E

'''conda activate RL101

  brew update && brew upgrade
  brew install ffmpeg
  brew install XQuartz
  brew install freeglut
  brew install cmake openmpi

  pip install 'stable-baselines3[extra]'
  conda install -c conda-forge box2d-py==2.3.8 
  # conda install -c conda-forge gym
  # pip install sb3-contrib
  ## pip install 'gym[all]'
'''
import gymnasium

env = gymnasium.make("LunarLander-v2")
env.reset()

# ---------------------------------------------------------
print("action space ::",  env.action_space)
print("sample action ::", env.action_space.sample())

print("obv space ::",       env.observation_space)
print("obv space shape ::", env.observation_space.shape)
print("obv sample ::",      env.observation_space.sample())

env.close()
