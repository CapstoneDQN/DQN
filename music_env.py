import numpy as np
import pandas as pd
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from collections import defaultdict

# 데이터 전처리 예시
TOP_K = 500        # consider only top-K frequent ids

# 1) Load CSV with correct format: user,id,timestamp
df = pd.read_csv(
    'dataset100000.csv', sep=',', header=0  # 첫 번째 행을 헤더로 처리
)

print("데이터 샘플:")
print(df.head())
print(f"음악 ID 샘플: {df['id'].head().tolist()}")

# 2) Limit to top-K frequent ids
id_counts = df['id'].value_counts()
top_items = id_counts.head(min(TOP_K, len(id_counts))).index.tolist()
df = df[df['id'].isin(top_items)]

# 3) Sort by user & timestamp, then aggregate ids per user
df = df.sort_values(['user','timestamp'])
user_groups = df.groupby('user')['id'].apply(list).reset_index()
user_groups.columns = ['user_id','listening_history']

# 4) Build item-index mappings for top-K
item2idx = {item: idx for idx, item in enumerate(top_items)}
idx2item = {idx: item for item, idx in item2idx.items()}
num_items = len(top_items)

# 5) Song Catalog - Matrix Factorization을 사용한 추천 시스템
class SongCatalog:
    def __init__(self, user_groups, top_items, n_factors=50):
        """
        Matrix Factorization을 사용한 음악 카탈로그
        
        Args:
            user_groups: 사용자별 청취 기록 데이터프레임
            top_items: 상위 인기 음악 리스트
            n_factors: latent factor 수 (차원)
        """
        self.user_groups = user_groups
        self.top_items = top_items
        self.item2idx = {item: idx for idx, item in enumerate(top_items)}
        self.n_factors = n_factors
        self.model = None
        self._build_matrix_and_train()
    
    def _build_matrix_and_train(self):
        """0/1 매트릭스 구성 및 Matrix Factorization 훈련"""
        # 1) 사용자-아이템 상호작용 데이터 생성 (0/1 matrix)
        interactions = []
        for _, row in self.user_groups.iterrows():
            user_id = row['user_id']
            listening_history = row['listening_history']
            
            # 각 사용자가 들은 음악에 대해 rating=1 부여
            for song_id in listening_history:
                if song_id in self.item2idx:
                    interactions.append((user_id, song_id, 1))  # (user, item, rating)
        
        # 2) Surprise 라이브러리용 데이터셋 생성
        df_interactions = pd.DataFrame(interactions, columns=['user', 'item', 'rating'])
        reader = Reader(rating_scale=(0, 1))  # 0/1 스케일
        data = Dataset.load_from_df(df_interactions[['user', 'item', 'rating']], reader)
        
        # 3) 훈련/테스트 분할
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
        
        # 4) SVD (Matrix Factorization) 모델 훈련
        self.model = SVD(n_factors=self.n_factors, random_state=42)
        self.model.fit(trainset)
    
    def get_recommendations(self, user_id, n_recommendations=10):
        """특정 사용자에게 음악 추천"""
        if self.model is None:
            raise ValueError("모델이 훈련되지 않았습니다.")
        
        # 사용자가 아직 듣지 않은 음악들에 대해 예측 점수 계산
        unheard_songs = []
        user_history = set()
        
        # 사용자 청취 기록 찾기
        user_row = self.user_groups[self.user_groups['user_id'] == user_id]
        if not user_row.empty:
            user_history = set(user_row.iloc[0]['listening_history'])
        
        # 듣지 않은 음악들에 대해 예측
        for song_id in self.top_items:
            if song_id not in user_history:
                predicted_rating = self.model.predict(user_id, song_id).est
                unheard_songs.append((song_id, predicted_rating))
        
        # 예측 점수 기준으로 정렬하여 상위 N개 추천
        unheard_songs.sort(key=lambda x: x[1], reverse=True)
        return [song_id for song_id, _ in unheard_songs[:n_recommendations]]

class idHistoryEnv(py_environment.PyEnvironment):
    def __init__(self, id_catalog, user_groups, song_catalog=None, max_steps=5):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32,
            minimum=0, maximum=len(id_catalog) - 1,
            name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(max_steps,), dtype=np.int32,
            minimum=0, maximum=TOP_K,  # 0부터 시작하도록 수정
            name='observation')
        self._id_catalog = id_catalog
        self._max_steps = max_steps
        self._history = np.zeros((max_steps,), dtype=np.int32)
        self._step_count = 0
        self._episode_ended = False

        # user_groups 저장
        self.user_groups = user_groups
        
        # Song Catalog 추가
        self.song_catalog = song_catalog

        # 사용자별 청취 시퀀스 리스트
        self.SEQ = self.user_groups['listening_history'].tolist()
        self.user_sequence = None

        # 현재 에피소드에서 참조할 사용자 인덱스 (초기화시 0)
        self.current_user_idx = 0
        self.current_user_id = None

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._history.fill(0)

        # 현재 에피소드에 사용할 사용자 선택 (충분한 시퀀스가 있는 사용자 선택)
        valid_users = [i for i, seq in enumerate(self.SEQ) if len(seq) > self._max_steps]
        if valid_users:
            self.current_user_idx = np.random.choice(valid_users)
        else:
            self.current_user_idx = 0
        
        # 현재 사용자 ID 저장
        self.current_user_id = self.user_groups.iloc[self.current_user_idx]['user_id']
        self.user_sequence = self.SEQ[self.current_user_idx]
        
        # 초기 상태: 첫 번째 아이템으로 시작
        if len(self.user_sequence) > 0 and self.user_sequence[0] in item2idx:
            self._history[0] = item2idx[self.user_sequence[0]]

        self._step_count = 0
        self._episode_ended = False

        return ts.restart(self._history.copy())
    
    def get_mf_recommendations(self, n_recommendations=10):
        """현재 사용자에 대한 Matrix Factorization 추천"""
        if self.song_catalog is None or self.current_user_id is None:
            return []
        
        return self.song_catalog.get_recommendations(self.current_user_id, n_recommendations)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        if not (0 <= action < len(self._id_catalog)):
            raise ValueError(f'Action {action} out of bounds.')

        # 다음 예상 아이템이 있는지 확인
        if self._step_count + 1 >= len(self.user_sequence):
            self._episode_ended = True
            return ts.termination(self._history.copy(), 0.0)

        # 추천된 노래 (action 인덱스를 실제 노래로 매핑)
        selected_id = self._id_catalog[action]
        expected_id = self.user_sequence[self._step_count + 1]

        # 보상 계산
        if selected_id == expected_id:
            reward = 1.0
        else:
            reward = 0.0

        # 상태 업데이트 (한 칸씩 시프트)
        prev = self._history.copy()
        self._history[:-1] = prev[1:]
        
        # 마지막에 실제로 선택된 아이템 추가 (expected_id 기준)
        if expected_id in item2idx:
            self._history[-1] = item2idx[expected_id]
        else:
            self._history[-1] = 0

        self._step_count += 1
        
        # 에피소드 종료 조건
        if self._step_count >= self._max_steps - 1 or self._step_count + 1 >= len(self.user_sequence):
            self._episode_ended = True
            return ts.termination(self._history.copy(), reward)

        # 중간 스텝
        return ts.transition(self._history.copy(), reward=reward, discount=1.0)


# 실행 예시
if __name__ == "__main__":
    # Song Catalog 생성 (Matrix Factorization)
    song_catalog = SongCatalog(user_groups, top_items, n_factors=50)
    
    # id_catalog을 top_items에서 선택하도록 수정
    id_catalog = top_items[:min(100, len(top_items))]
    env = idHistoryEnv(id_catalog, user_groups, song_catalog=song_catalog)

    state = env.reset()
    print(f"초기 상태: {state}")
    
    mf_recommendations = env.get_mf_recommendations(5)
    print(f"MF 추천: {mf_recommendations}")
    
    transition = env.step(5)
    print(f"전환 결과: {transition}")
