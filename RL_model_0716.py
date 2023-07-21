import gym
from stable_baselines3 import PPO
# from stable_baselines3.common.envs import DummyVecEnv
# from stable_baselines3.common.env_checker import check_env

MAX_STEPS = 10000  # 12hr * 60min * 60sec / 7초당 1회 => 6171
MAX_QUEUED_COMMAND = 200
ACTION_VALUE_RANGE = 30  # action 값 최대

# OHT Routing을 강화학습 환경으로 구현한 클래스
class RoutingEnv(gym.Env):
    def __init__(self, max_steps, pclient):
        print('RoutingEnv.__init__()')
        super().__init__()

        # pclient 데이터 초기화 및 저장
        self.pclient = pclient
        self.current_step = 0
        # 학습에 사용할 최대 스텝 수 
        self.max_steps = max_steps

        # observation space 설정
        observation_shape = 4 + pclient.RAILINE_COUNT * 8
        # self.observation_space = gym.spaces.Box(low=0, high=float('inf'), shape=(33,), dtype=float)
        self.observation_space = gym.spaces.Box(low=0, high=float('inf'), shape=(observation_shape,), dtype=float)  
        # action space 설정
        # self.action_space = gym.spaces.Discrete(pclient.RAILLINE_COUNT)
        self.action_space = gym.spaces.Box(low=0, high=ACTION_VALUE_RANGE, shape=(pclient.RAILINE_COUNT,), dtype=float) 

    def reset(self):
        print('RoutingEnv.reset()')
        # 초기화
        self.current_step = 0
        observation = self._get_observation()
        return observation

    # step() 실행 전에 _set_state(), _set_cost() 실행 필요
    def step(self, action):
        print('RoutingEnv.step()')
        # action 실행
        # self._execute_action(action)

        # 다음 step 진행
        self.current_step += 1

        # 종료 조건 확인
        done = ( self.current_step >= self.max_steps or self.pclient.QueuedCommandCount > MAX_QUEUED_COMMAND ) 
        if done:
            info = {'episode': {'status': 'Time limit reached'}}
        else:
            info = {}

        # observation, reward, done, info 반환
        observation = self._get_observation()
        reward = self._get_reward()

        return observation, reward, done, info

    def _set_state(self, pclient, railline_oht_state):
        print('RoutingEnv.__set_status()')
        self.pclient = pclient
        self.railline_oht_state = railline_oht_state

    def _get_observation(self):
        print('RoutingEnv._get_observation()')
        # self.pclient = pclient

        # 현재 상태(observation) 생성
        observation = []

        observation.extend([
            self.pclient.CompletedCommandCount / 1000,
            self.pclient.TransferCommandCount / 1000,
            self.pclient.WaitingCommandCount / 1000,
            self.pclient.QueuedCommandCount / 1000
        ])
        
        # pclient.RAILLINE_DIC 정보
        for _, railLine in self.pclient.RAILLINE_DIC.items():
            observation.extend([
                railLine.DistancePerVelocity,
                railLine.Distance / 1000,
                railLine.LevelJoiningLineCount,
                railLine.DivergingLineCount
            ])

        # railline 별 workingOHTs, movingOHTs, idleOHTs, reservedOHTs
        for _, oht_state in self.railline_oht_state.items():
            observation.extend([
                oht_state['working'],
                oht_state['moving'],
                oht_state['idle'],
                oht_state['reserved']
            ])

        return observation

    def _execute_action(self, action):
        print('RoutingEnv._execute_action()')
        # action 실행
        # cost = 1  # 예시로 임의의 cost 값 설정

        # pclient.RAILLINECOST_DIC 업데이트
        # self.pclient.RAILLINECOST_DIC[action]['FRailLineCost'] = cost
        for cost in action:
            self.pclient.RAILLINECOST_DIC[action]['FRailLineCost'] = cost

    def _set_cost(self, cost_before, cost_after):
        self.cost_before = cost_before
        self.cost_after = cost_after

    def _get_reward(self):
        print('RoutingEnv._get_reward()')
        # reward 계산
        # old_total_cost = sum(railline['FRailLineCost'] for railline in self.pclient.RAILLINECOST_DIC.values())
        total_cost_before = sum(railline.FRailLineCost for railline in self.cost_before.values())

        # 이전 step의 total cost와 현재 step의 total cost 비교
        # current_total_cost = sum(railline['FRailLineCost'] for railline in raillinecost_dic.values())
        total_cost_after = sum(railline.FRailLineCost for railline in self.cost_after.values())

        if total_cost_before == 0:  # the first step
            reward = 0
        else:
            if total_cost_after == total_cost_before:
                reward = 0
            elif total_cost_after < total_cost_before:
                reward = 1
            else:
                reward = -1

        return reward


# pclient 데이터로 SimEnv 생성
# env = RoutingEnv(MAX_STEPS, pclient)

# PPO 모델 초기화
# model = PPO("MlpPolicy", env, verbose=1)

# 학습 시작
# model.learn(total_timesteps=MAX_STEPS)

# 학습된 모델 저장
# model.save("trained_model")
