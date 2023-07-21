#import gymnasium as gym
import gym
from stable_baselines3 import A2C


# Carpole 환경 생성
env = gym.make('CartPole-v1')

class TetrisApp(gym.Env):
	def __init__(self):
    	self.height = 20
        self.width = 10
        self.observation_space = spaces.Box(low=0, high=1e+8, shape=(1, 4), dtype=np.float)
        self.action_space = spaces.Discrete(5)






# env = gym.make("CartPole-v1", render_mode="rgb_array")

# 환경 초기화
observation = env.reset()

# 에피소드 수행
for t in range(1000):
    # 환경 시각화
    env.render()

    # 행동 선택 (예시: 왼쪽으로 가속)
    action = 0

    # 선택한 행동 실행
    observation, reward, done, info = env.step(action)

    # 에피소드 종료 여부 확인
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        break

# 환경 종료
env.close()



#---------------------------------------------------------------------
model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)
# model = A2C("MlpPolicy", "CartPole-v1").learn(10000)


vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()