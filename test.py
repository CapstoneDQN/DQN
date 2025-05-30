import numpy as np
import pandas as pd
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# -------------------- 데이터 전처리 --------------------
TOP_K = 500  # 상위 K개 곡만 사용

df = pd.read_csv('listening_history.csv', sep=r'\s+', header=None, names=['user', 'song', 'date', 'time'])
df['timestamp'] = df['date'] + ' ' + df['time']

# 상위 곡 필터링
song_counts = df['song'].value_counts()
top_items = song_counts.head(TOP_K).index.tolist()
df = df[df['song'].isin(top_items)]

# 사용자별 시퀀스 생성
df = df.sort_values(['user', 'timestamp'])
user_groups = df.groupby('user')['song'].apply(list).reset_index()
user_groups.columns = ['user_id', 'listening_history']

item2idx = {item: idx for idx, item in enumerate(top_items)}
idx2item = {idx: item for item, idx in item2idx.items()}
num_items = len(top_items)

# -------------------- Surprise 기반 추천 시스템 --------------------
class SongCatalog:
    def __init__(self, user_groups, top_items, n_factors=50):
        self.user_groups = user_groups
        self.top_items = top_items
        self.item2idx = {item: idx for idx, item in enumerate(top_items)}
        self.n_factors = n_factors
        self.model = None
        self._train()

    def _train(self):
        interactions = []
        for _, row in self.user_groups.iterrows():
            user_id = row['user_id']
            for song_id in row['listening_history']:
                if song_id in self.item2idx:
                    interactions.append((user_id, song_id, 1))
        df_interactions = pd.DataFrame(interactions, columns=['user', 'item', 'rating'])
        reader = Reader(rating_scale=(0, 1))
        data = Dataset.load_from_df(df_interactions[['user', 'item', 'rating']], reader)
        trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
        self.model = SVD(n_factors=self.n_factors, random_state=42)
        self.model.fit(trainset)

    def get_recommendations(self, user_id, n=5):
        user_row = self.user_groups[self.user_groups['user_id'] == user_id]
        if user_row.empty:
            return []
        user_history = set(user_row.iloc[0]['listening_history'])

        scores = []
        for song_id in self.top_items:
            if song_id not in user_history:
                pred = self.model.predict(user_id, song_id).est
                scores.append((song_id, pred))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scores[:n]]
# 한 사용자에 대해 나온 추천 노래가 실제로 들어가 있는지?

# -------------------- 강화학습 환경 정의 --------------------
class SongHistoryEnv(py_environment.PyEnvironment):
    def __init__(self, song_catalog, user_groups, max_steps=5):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=len(song_catalog.top_items) - 1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(max_steps,), dtype=np.int32, minimum=1, maximum=TOP_K, name='observation')

        self._song_catalog = song_catalog
        self._max_steps = max_steps
        self._history = np.zeros((max_steps,), dtype=np.int32)
        self._step_count = 0
        self._episode_ended = False

        self.user_groups = user_groups
        self.SEQ = self.user_groups['listening_history'].tolist()
        self.user_sequence = None
        self.current_user_idx = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._history.fill(0)
        self.current_user_idx = np.random.randint(len(self.SEQ))
        self.user_sequence = self.SEQ[self.current_user_idx]
        self._step_count = 0
        self._episode_ended = False
        return ts.restart(self._history)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        if not (0 <= action < len(self._song_catalog.top_items)):
            raise ValueError(f'Action {action} out of bounds.')

        selected_song = self._song_catalog.top_items[action]

        if selected_song in self.user_sequence[self._step_count + 1:]:
            reward = 1.0
        else:
            reward = 0.0

        prev = self._history.copy()
        self._history[:-1] = prev[1:]
        if self._step_count + 1 < len(self.user_sequence):
            next_song = self.user_sequence[self._step_count + 1]
            self._history[-1] = item2idx.get(next_song, 0) + 1
        else:
            self._history[-1] = 0  # 또는 padding 값
        
        self._step_count += 1
        if self._step_count >= self._max_steps - 1:
            self._episode_ended = True
            return ts.termination(np.array([self._history], dtype=np.int32), reward)

        return ts.transition(np.array([self._history], dtype=np.int32), reward=reward, discount=1.0)

    def get_mf_recommendations(self, n=5):
        current_user_id = self.user_groups.iloc[self.current_user_idx]['user_id']
        # 현재 스텝까지의 청취 이력만 사용
        current_history = set(self.user_sequence[:self._step_count + 1])
        
        scores = []
        for song_id in self._song_catalog.top_items:
            if song_id not in current_history:  # 현재까지 들은 곡만 제외
                pred = self._song_catalog.model.predict(current_user_id, song_id).est
                scores.append((song_id, pred))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scores[:n]]

# -------------------- 실행 --------------------
if __name__ == "__main__":
    print("Surprise SVD 학습 중...")
    song_catalog = SongCatalog(user_groups, top_items)
    print("SVD 학습 완료")

    env = SongHistoryEnv(song_catalog, user_groups)

    print("\n환경 초기화")
    state = env.reset()
    print("초기 상태:", state.observation)

    total_reward = 0

    for step_num in range(env._max_steps - 1):
        # Surprise SVD 추천곡 후보 10개 가져오기
        recommended_songs = env.get_mf_recommendations(n=10)
        if not recommended_songs:
            # 추천곡 없으면 랜덤 액션
            action = np.random.randint(len(song_catalog.top_items))
            chosen_song = song_catalog.top_items[action]
        else:
            # 추천 후보 중에서 랜덤으로 하나 선택
            chosen_song = recommended_songs[np.random.randint(len(recommended_songs))]
            action = item2idx[chosen_song]

        time_step = env.step(action)
        total_reward += time_step.reward

        print(f"\nStep {step_num + 1}")
        print("추천곡 ID:", chosen_song)
        print("액션 인덱스:", action)
        print("보상:", time_step.reward)
        print("상태:", time_step.observation)

        if time_step.is_last():
            print("에피소드 종료")
            break

    print("\n총 누적 보상:", total_reward)
