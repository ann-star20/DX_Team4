# AI 모델을 위해 파이썬이 설치된 리눅스 베이스 이미지 사용
FROM python:3.10-slim
ENV TZ=Asia/Seoul

# 시스템 의존성 설치 (pyaudio, librosa 등을 위해 필요)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    portaudio19-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 라이브러리 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# 실행 (0.0.0.0 필수!)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]