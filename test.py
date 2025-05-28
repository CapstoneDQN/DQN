import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class SongHistoryEnv(py_environment.PyEnvironment):
    def __init__(self, song_catalog, max_steps=5):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32,
            minimum=0, maximum=len(song_catalog) - 1,


            name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(max_steps,), dtype=np.int32,
            minimum=0, maximum=1,
            name='observation')
        self._song_catalog = song_catalog
        self._max_steps = max_steps
        self._history = np.zeros((max_steps,), dtype=np.int32)
        self._step_count = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        # [1, 0, 0, 0, 0] 형태로 초기화
        self._history.fill(0)
        self._history[0] = 1
        self._step_count = 0
        self._episode_ended = False
        return ts.restart(self._history)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        if not (0 <= action < len(self._song_catalog)):
            raise ValueError(f'Action {action} out of bounds.')

        # 1) 한 칸 시프트
        prev = self._history.copy()
        self._history[:-1] = prev[1:]
        # 2) 마지막 슬롯: 이전 윈도우에 action이 있으면 1, 없으면 0
        self._history[-1] = 1 if action in prev else 0

        # 스텝 증가 및 종료 판단
        self._step_count += 1
        if self._step_count >= self._max_steps:
            self._episode_ended = True
            # 종료 시 보상 = 총 스텝 수, Last 단계계
            return ts.termination(self._history, float(self._step_count))

        # 중간 단계 : 보상 0, 할인 1.0
        return ts.transition(self._history, reward=0.0, discount=1.0)

'''
1. _reset()

모든 슬롯을 0으로 채우고

self._history[0] = 1 (첫 번째만 1)

ts.restart(self._history)로 반환

2. _step(action)

이전 윈도우를 복사해 한 칸씩 이동

마지막에 action in prev 여부로 1/0 채움

스텝 수가 max_steps에 도달하면 ts.termination()

그 외엔 ts.transition()'''


다시 이 코드의 설명을 자세히 해줘.
