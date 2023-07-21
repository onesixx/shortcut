import gym
from stable_baselines3 import PPO
# from stable_baselines3.common.envs import DummyVecEnv
# from stable_baselines3.common.env_checker import check_env
import redis
import copy
import numpy as np
from datetime import datetime
# import time

# 상수값
MAX_STEPS = 10000  # 12hr * 60min * 60sec / 7초당 1회 => 6171
MAX_QUEUED_COMMAND = 200
ACTION_VALUE_RANGE = 30  # action 값 최대
LISTEN_TIMEOUT = 60  # 60 sec
RAILINE_COUNT = 6454
TENSORBOARD_LOG = './model/log/'

# OHT Routing을 강화학습 환경으로 구현한 클래스
class RoutingEnv(gym.Env):
    def __init__(self):
        print('RoutingEnv.__init__()')
        super().__init__()

        self.current_step = 0

        # observation space 설정
        self.observation_shape = 4 + RAILINE_COUNT * 8
        self.observation_space = gym.spaces.Box(low=0, high=float('inf'), shape=(self.observation_shape,), dtype=float)  

        # action space 설정
        self.action_space = gym.spaces.Box(low=0, high=ACTION_VALUE_RANGE, shape=(RAILINE_COUNT,), dtype=float) 

        # ClientAlgorithm()과 통신할 Message Queue 선언
        self.rq = redis.Redis(host='localhost', port=6379, decode_responses=True)

        self.state = {}
        self.cost_after_action = {}

    def reset(self):
        print('RoutingEnv.reset()')
        # 초기화
        self.current_step = 0
        # observation = self._get_observation()
        observation = self._init_observation()
        return observation

    def step(self, action):
        print('RoutingEnv.step()')

        # 다음 step 진행
        self.current_step += 1

        # get observation
        observation = self._get_observation()

        # 종료 조건 확인
        if self.current_step >= MAX_STEPS:
            info = {'episode': {'status': 'Learning step limit reached'}}
            done = True
        elif self.state.QueuedCommandCount > MAX_QUEUED_COMMAND:
            info = {'episode': {'status': 'QueuedCommandCount limit reached'}}
            done = True
        else:
            info = {}
            done = False

        # action 실행
        self._execute_action(action)

        # reward 계산
        reward = self._get_reward()

        # observation, reward, done, info 반환
        return observation, reward, done, info

    def _init_observation(self):
        return np.zeros(self.observation_shape)

    def receive_rq(self, message_name):
        p = self.rq.pubsub()
        p.subscribe(message_name)

        # 데이터 수신 대기
        for message in p.listen():
            if message['type'] == 'message':
                data = message['data']
                print('receive_data(): success')

            if not p.subscribed:
                break

        return data

    def _get_observation(self):
        print('RoutingEnv._get_observation()')

        self.state = self.receive_rq("J-state")

        # 현재 상태(observation) 생성
        observation = []

        observation.extend([
            self.state.CompletedCommandCount / 1000,
            self.state.TransferCommandCount / 1000,
            self.state.WaitingCommandCount / 1000,
            self.state.QueuedCommandCount / 1000
        ])
        
        # state.RAILLINE_DIC 정보
        for _, railLine in self.state.RAILLINE_DIC.items():
            observation.extend([
                railLine.DistancePerVelocity,
                railLine.Distance / 1000,
                railLine.LevelJoiningLineCount,
                railLine.DivergingLineCount
            ])

        # railline 별 workingOHTs, movingOHTs, idleOHTs, reservedOHTs
        for _, oht_state in self.state.RAILLINE_OHT_STATE.items():
            observation.extend([
                oht_state['working'],
                oht_state['moving'],
                oht_state['idle'],
                oht_state['reserved']
            ])

        return observation

    def _execute_action(self, action):
        # action 실행
        print('RoutingEnv._execute_action()')
        
        #### random으로 OHT 상태값 기준으로 계산한 cost를 사용하는 방안 고려

        self.cost_after_action = copy.deepcopy(self.state.RAILLINECOST_DIC)

        # for id, cost in action.items():
        for id, cost in action[0].items():
            self.cost_after_action[id+1]['FRailLineCost'] = cost

        # ClientAlgorithm에 action데이터(cost) 전달
        self.rq.publish('J-action', self.cost_after_action)

    def _get_reward(self):
        print('RoutingEnv._get_reward()')
        # reward 계산
        total_cost_before = sum(railline['FRailLineCost'] for railline in self.state.RAILLINECOST_DIC.values())

        # 이전 step의 total cost와 현재 step의 total cost 비교
        total_cost_after = sum(railline.FRailLineCost for railline in self.cost_after_action.values())

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


if __name__ == "__main__":
    # RoutingEnv 생성
    env = RoutingEnv()

    # PPO 모델 초기화
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=TENSORBOARD_LOG)

    # 학습 시작
    model.learn(total_timesteps=MAX_STEPS)

    # 학습된 모델 저장
    time_str = datetime.now().strftime('%Y%m%d%H%H%S%f')
    model.save(f'./model/{time_str}')
