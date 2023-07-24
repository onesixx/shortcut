

from snake_env import SnekEnv
from stable_baselines3.common.env_checker import check_env

env = SnekEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)