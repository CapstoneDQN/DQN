import numpy as np
import pandas as pd
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

# 데이터 전처리 예시
df = pd.read_csv('listening_history.csv')  # 컬럼: user, song, timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values(['user', 'timestamp'])
user_groups = df.groupby('user')['song'].apply(list).reset_index()
user_groups.columns = ['user_id', 'listening_history']

class SongHistoryEnv(py_environment.PyEnvironment):
    def __init__(self, song_catalog, user_groups, max_steps=5):
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

        # user_groups 저장
        self.user_groups = user_groups

        # 사용자별 청취 시퀀스 리스트
        self.SEQ = self.user_groups['listening_history'].tolist()

        # 현재 에피소드에서 참조할 사용자 인덱스 (초기화시 0)
        self.current_user_idx = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._history.fill(0)

        # 현재 에피소드에 사용할 사용자 인덱스 초기화 (무작위 또는 순차 선택 가능)
        self.current_user_idx = 0  # 필요시 랜덤 선택도 가능: np.random.randint(len(self.SEQ))

        # 초기 상태: 최근 최대 max_steps까지 청취 기록 반영 (여기서는 단순히 첫 인덱스만 1로 세팅)
        if len(self.SEQ[self.current_user_idx]) > 0:
            self._history[0] = 1

        self._step_count = 0
        self._episode_ended = False

        return ts.restart(self._history)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        if not (0 <= action < len(self._song_catalog)):
            raise ValueError(f'Action {action} out of bounds.')

        prev = self._history.copy()
        # 한 칸 시프트 (과거 상태 이동)
        self._history[:-1] = prev[1:]

        # 추천된 노래 (action 인덱스를 실제 노래로 매핑)
        selected_song = self._song_catalog[action]

        # 현재 사용자 청취 시퀀스 참조
        current_seq = self.SEQ[self.current_user_idx]

        # 마지막 슬롯에 추천곡이 현재 사용자 청취 시퀀스에 있는지 반영
        self._history[-1] = 1 if selected_song in current_seq else 0
        
        if selected_song in current_seq:
            # 추천곡이 청취 기록에 있다면 보상 1, 아니면 0
            reward = 1.0
        else:
            reward = 0.0

        self._step_count += 1
        if self._step_count >= self._max_steps:
            self._episode_ended = True
            # 종료 시 보상 등은 여기서 추가 구현 가능
            return ts.termination(self._history, float(self._step_count))

        # 중간 스텝에서는 보상 0, 할인율 1.0 반환
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


