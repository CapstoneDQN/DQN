import tensorflow as tf
import time

print("=== TensorFlow GPU 테스트 ===")
print(f"TensorFlow 버전: {tf.__version__}")

# 1. 사용 가능한 GPU 디바이스 확인
print("\n1. GPU 디바이스 목록:")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f"   - {gpu}")
    print(f"   총 {len(gpus)}개의 GPU 발견")
else:
    print("   GPU를 찾을 수 없습니다")

# 2. 논리적 디바이스 확인
print("\n2. 논리적 디바이스 목록:")
devices = tf.config.list_logical_devices()
for device in devices:
    print(f"   - {device}")

# 3. TensorFlow가 CUDA를 사용할 수 있는지 확인
print(f"\n3. CUDA 지원 여부: {tf.test.is_built_with_cuda()}")
print(f"4. GPU 사용 가능 여부: {tf.test.is_gpu_available()}")

# 5. 간단한 연산 테스트 (CPU vs GPU)
print("\n5. 연산 성능 테스트:")

# 큰 행렬 생성
matrix_size = 5000
a = tf.random.normal([matrix_size, matrix_size])
b = tf.random.normal([matrix_size, matrix_size])

# CPU에서 연산
print("   CPU 연산 중...")
with tf.device('/CPU:0'):
    start_time = time.time()
    c_cpu = tf.matmul(a, b)
    cpu_time = time.time() - start_time
    print(f"   CPU 시간: {cpu_time:.4f}초")

# GPU에서 연산 (사용 가능한 경우)
if gpus:
    print("   GPU 연산 중...")
    with tf.device('/GPU:0'):
        start_time = time.time()
        c_gpu = tf.matmul(a, b)
        gpu_time = time.time() - start_time
        print(f"   GPU 시간: {gpu_time:.4f}초")
        
        if cpu_time > gpu_time:
            print(f"   GPU가 {cpu_time/gpu_time:.2f}배 더 빠릅니다!")
        else:
            print("   CPU와 GPU 성능 차이가 크지 않거나 GPU가 더 느립니다.")
else:
    print("   GPU를 사용할 수 없습니다.")

# 6. GPU 메모리 사용량 확인
if gpus:
    print("\n6. GPU 메모리 정보:")
    try:
        for i, gpu in enumerate(gpus):
            memory_info = tf.config.experimental.get_memory_info(f'GPU:{i}')
            print(f"   GPU {i} 현재 메모리 사용량: {memory_info['current'] / (1024**3):.2f} GB")
            print(f"   GPU {i} 최대 메모리 사용량: {memory_info['peak'] / (1024**3):.2f} GB")
    except:
        print("   GPU 메모리 정보를 가져올 수 없습니다.")

print("\n=== 테스트 완료 ===")

# PR용 테스트 주석