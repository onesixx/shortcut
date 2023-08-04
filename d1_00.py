# Agent : 자동차
# observation_space : Agent가 환경을 볼 수 있는 범위를 의미. 해당 공간에서만 정보를 얻을 수 있습니다. 이 관찰공간에서 low는 x축 좌표의 최솟값과, 최소 속도입니다. high는 x축 좌표의 최댓값과 최대 속도를 보여줍니다.
# max_episode_steps : 각 에피소드마다의 종료 조건을 의미합니다.해당 값이 n이라면, 최대 n번의 time step을 가지고 n번 움직이면 종료된다는 것을 의미합니다.


# Action : '왼쪽', '정지', '오른쪽'

import gymnasium
env = gymnasium.make('MountainCar-v0')

print(env.observation_space)
print(env.observation_space.low)
print(env.observation_space.high)
print(env._max_episode_steps)

print(env.action_space)