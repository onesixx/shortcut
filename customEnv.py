
import gym

MAX_STEPS = 5  # 12hr * 60min * 60sec / 7초당 1회 => 6171
ACTION_VALUE_RANGE = 30  # action 값 최대
TENSORBOARD_LOG = './model/log/'

# OHT Routing을 강화학습 환경으로 구현한 클래스

class customEnv(gym.Env):
    def __init__(self):
        print('my env __init__()')
        super().__init__()

        self.current_step = 0

        # observation space 설정
        self.observation_space = gym.spaces.Box(low=0, high=float('inf'), shape=(4,), dtype=float)

        # action space 설정
        self.action_space = gym.spaces.Box(low=0, high=ACTION_VALUE_RANGE, shape=(1,), dtype=float)


    def reset(self):
        print('my Env reset()')
        # 초기화
        self.current_step = 0
        observation = self._get_observation()
        return observation

    # step() 실행 전에 _set_state(), _set_cost() 실행 필요
    def step(self, action):
        print('my Env step()')
        # action 실행
        # self._execute_action(action)

        # 다음 step 진행
        self.current_step += 1

        # observation, reward, done, info 반환
        observation = self._get_observation()
        reward = self._get_reward()
        done = False
        info = {}

        return observation, reward, done, info

    def _get_observation(self):
        print('my Env._get_observation()')
        # self.pclient = pclient
        observation = [1, 2, 3, 4, 5]
        return observation

    def _execute_action(self, action):
        print('my Env _execute_action()')
        # action 실행

    def _get_reward(self):
        print('my Env _get_reward()')
        # reward 계산

        reward = 1
        return reward