# 서버 GPU 드라이버에 맞는 CUDA 런타임 이미지 사용
# (예: 서버 CUDA 12.1 드라이버 → nvidia/cuda:12.1.1-runtime-ubuntu22.04)
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
 && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN python3 -m pip install --upgrade pip

# requirements.txt 있으면 먼저 복사해서 설치 (빌드 캐시 최적화)
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 기본 실행 명령
CMD ["bash"]
