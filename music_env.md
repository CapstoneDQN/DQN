## 1. 데이터 전처리

```python
# CSV 파일 로드
df = pd.read_csv('dataset100000.csv', sep=r'\s+', header=None, names=['user','id','date','time'])

# 타임스탬프 생성 (타입 오류 방지)
df['timestamp'] = df['date'].astype(str) + ' ' + df['time'].astype(str)

# 상위 K개 인기 음악 선별
id_counts = df['id'].value_counts()
top_items = id_counts.head(min(TOP_K, len(id_counts))).index.tolist()
df = df[df['id'].isin(top_items)]

# 사용자별 청취 기록 집계
df = df.sort_values(['user','timestamp'])
user_groups = df.groupby('user')['id'].apply(list).reset_index()
user_groups.columns = ['user_id','listening_history']
```

**설명**:
- 공백으로 구분된 CSV 파일을 읽어 사용자별 음악 청취 이력 생성
- `.astype(str)`: float와 string 타입 충돌 방지
- 인기도 기반 필터링으로 노이즈 제거 및 성능 향상
- 시간순 정렬 후 사용자별로 청취한 음악 리스트 생성

## 2. 음악-인덱스 매핑

```python
# 음악 ID를 인덱스로 매핑 (0부터 시작)
item2idx = {item: idx for idx, item in enumerate(top_items)}
idx2item = {idx: item for item, idx in item2idx.items()}
num_items = len(top_items)
```

**설명**:
- 문자열/숫자 ID를 0부터 시작하는 연속된 인덱스로 변환
- 배열 접근 및 신경망 임베딩에서 효율적 처리를 위함

## 3. SongCatalog 클래스 - Matrix Factorization

### 3.1 초기화 및 모델 훈련

```python
class SongCatalog:
    def __init__(self, user_groups, top_items, n_factors=50):
        self.user_groups = user_groups
        self.top_items = top_items
        self.item2idx = {item: idx for idx, item in enumerate(top_items)}
        self.n_factors = n_factors  # latent factor 차원
        self.model = None
        self.user_item_matrix = None
        self._build_matrix_and_train()
```

**설명**:
- `n_factors`: SVD 분해 시 사용할 잠재 요인(latent factor) 수
- 초기화와 동시에 Matrix Factorization 모델 훈련 실행

### 3.2 0/1 매트릭스 구성 및 SVD 훈련

```python
def _build_matrix_and_train(self):
    # 1) 사용자-아이템 상호작용 데이터 생성 (0/1 matrix)
    interactions = []
    for _, row in self.user_groups.iterrows():
        user_id = row['user_id']
        listening_history = row['listening_history']
        
        # 각 사용자가 들은 음악에 대해 rating=1 부여
        for song_id in listening_history:
            if song_id in self.item2idx:
                interactions.append((user_id, song_id, 1))
    
    # 2) Surprise 라이브러리용 데이터셋 생성
    df_interactions = pd.DataFrame(interactions, columns=['user', 'item', 'rating'])
    reader = Reader(rating_scale=(0, 1))  # 0/1 스케일
    data = Dataset.load_from_df(df_interactions[['user', 'item', 'rating']], reader)
    
    # 3) SVD 모델 훈련
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
    self.model = SVD(n_factors=self.n_factors, random_state=42)
    self.model.fit(trainset)
```

**설명**:
- **0/1 Matrix**: 사용자가 음악을 들었으면 1, 안 들었으면 0
- **SVD**: 사용자×음악 행렬을 사용자 특성×잠재요인, 잠재요인×음악 특성으로 분해
- 훈련/테스트 분할로 모델 성능 검증 가능

### 3.3 개인화 추천

```python
def get_recommendations(self, user_id, n_recommendations=10):
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
```

**설명**:
- 사용자가 이미 들은 음악은 제외하고 새로운 음악만 추천
- SVD 모델로 예측한 선호도 점수 기준으로 정렬
- 상위 N개 음악의 ID만 반환

### 3.4 음악 유사도 계산

```python
def get_song_similarity(self, song_id, n_similar=5):
    # 아이템 latent factors 추출
    item_factors = self.model.qi  # 아이템 factor 매트릭스
    song_idx = self.item2idx[song_id]
    song_vector = item_factors[song_idx]
    
    # 코사인 유사도 계산
    similarities = []
    for other_song_id in self.top_items:
        if other_song_id != song_id:
            other_idx = self.item2idx[other_song_id]
            other_vector = item_factors[other_idx]
            
            # 코사인 유사도 공식
            similarity = np.dot(song_vector, other_vector) / (
                np.linalg.norm(song_vector) * np.linalg.norm(other_vector)
            )
            similarities.append((other_song_id, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [song_id for song_id, _ in similarities[:n_similar]]
```

**설명**:
- `model.qi`: SVD에서 학습된 아이템(음악)의 잠재 특성 벡터
- **코사인 유사도**: 두 벡터 간 각도로 유사성 측정
- 특정 음악과 비슷한 특성을 가진 다른 음악들 찾기

## 4. idHistoryEnv 클래스 - 강화학습 환경

### 4.1 초기화

```python
class idHistoryEnv(py_environment.PyEnvironment):
    def __init__(self, id_catalog, user_groups, song_catalog=None, max_steps=5):
        # 액션 스펙: 추천할 수 있는 음악의 범위
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32,
            minimum=0, maximum=len(id_catalog) - 1,
            name='action')
        
        # 관찰 스펙: 최근 청취한 음악들의 시퀀스
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(max_steps,), dtype=np.int32,
            minimum=0, maximum=TOP_K,
            name='observation')
        
        self._id_catalog = id_catalog  # 추천 가능한 음악 목록
        self._max_steps = max_steps    # 에피소드 길이
        self._history = np.zeros((max_steps,), dtype=np.int32)  # 상태
        self.song_catalog = song_catalog  # Matrix Factorization 모델
```

**설명**:
- **action_spec**: 에이전트가 선택할 수 있는 행동(음악 인덱스) 범위
- **observation_spec**: 상태 공간 정의 (최근 5개 음악의 인덱스)
- `song_catalog`: MF 추천을 강화학습과 결합하기 위한 참조

### 5.2 에피소드 초기화

```python
def _reset(self):
    self._history.fill(0)  # 모든 슬롯을 0으로 초기화
    
    # 충분한 길이의 청취 기록을 가진 사용자 선택
    valid_users = [i for i, seq in enumerate(self.SEQ) if len(seq) > self._max_steps]
    if valid_users:
        self.current_user_idx = np.random.choice(valid_users)
    else:
        self.current_user_idx = 0
    
    # 현재 사용자 정보 설정
    self.current_user_id = self.user_groups.iloc[self.current_user_idx]['user_id']
    self.user_sequence = self.SEQ[self.current_user_idx]
    
    # 첫 번째 음악으로 초기 상태 설정
    if len(self.user_sequence) > 0 and self.user_sequence[0] in item2idx:
        self._history[0] = item2idx[self.user_sequence[0]]
    
    return ts.restart(self._history.copy())
```

**설명**:
- 에피소드마다 다른 사용자의 청취 이력을 사용
- 충분한 데이터가 있는 사용자만 선택하여 학습 품질 보장
- 첫 번째 음악을 상태의 시작점으로 설정

### 5.3 액션 실행 및 상태 전환

```python
def _step(self, action):
    # 추천한 음악 vs 실제로 들은 음악 비교
    selected_id = self._id_catalog[action]  # 에이전트가 추천한 음악
    expected_id = self.user_sequence[self._step_count + 1]  # 실제로 들은 음악
    
    # 보상 계산: 맞으면 1, 틀리면 0
    if selected_id == expected_id:
        reward = 1.0
    else:
        reward = 0.0
    
    # 상태 업데이트: 슬라이딩 윈도우 방식
    prev = self._history.copy()
    self._history[:-1] = prev[1:]  # 한 칸씩 앞으로 이동
    
    # 마지막에 실제로 들은 음악 추가
    if expected_id in item2idx:
        self._history[-1] = item2idx[expected_id]
    else:
        self._history[-1] = 0
    
    self._step_count += 1
    
    # 에피소드 종료 조건 확인
    if self._step_count >= self._max_steps - 1 or self._step_count + 1 >= len(self.user_sequence):
        self._episode_ended = True
        return ts.termination(self._history.copy(), reward)
    
    return ts.transition(self._history.copy(), reward=reward, discount=1.0)
```

**설명**:
- **슬라이딩 윈도우**: `[a,b,c,d,e]` → `[b,c,d,e,f]` 방식으로 상태 업데이트
- **보상 시스템**: 추천 정확도만 평가 (정확하면 1, 틀리면 0)
- **에피소드 종료**: 최대 스텝 수 도달 또는 사용자 시퀀스 끝

### 5.4 MF 추천 연동

```python
def get_mf_recommendations(self, n_recommendations=10):
    """현재 사용자에 대한 Matrix Factorization 추천"""
    if self.song_catalog is None or self.current_user_id is None:
        return []
    
    return self.song_catalog.get_recommendations(self.current_user_id, n_recommendations)
```

**설명**:
- 강화학습 환경에서 Matrix Factorization 추천을 받을 수 있는 인터페이스
- 두 방법론을 결합한 하이브리드 추천 시스템 구현 가능
