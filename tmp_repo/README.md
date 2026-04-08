
### 실행 명령어
프로젝트 루트
```# 단일 문장 추론
python scripts/text_infer_cli.py --text "오늘 기분이 좋아"

# 대화형 모드
python scripts/text_infer_cli.py

# 커스텀 체크포인트 사용
python scripts/text_infer_cli.py --ckpt /path/to/checkpoint.pt```

###  체크포인트 파일 선택지
코드는 자동으로 다음 순서로 체크포인트를 탐색
- best_kcelectra_lora.pt ← 권장 (빠름)
- models/text/best/ (HF/PEFT 디렉터리)
- models/text/last/ (HF/PEFT 디렉터리)
- --ckpt 옵션으로 직접 지정

### 음성 테스트
python3 scripts/audio_infer_cli.py          # 마이크 반복 모드
python3 scripts/audio_infer_cli.py --mic    # 마이크 1회
python3 scripts/audio_infer_cli.py --wav "path/to.wav"
무음이 너무 길거나 너무 짧으면 전처리 실패(duration) 같은 메시지를 띄우고 추론을 건너뜁니다.

python scripts/audio_infer_cli.py --wav "sample.wav"

uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
uvicorn app.main:app --host 127.0.0.1 --port 7000 --reload
uvicorn app.main:app --host 127.0.0.1 --port 7001 --reload
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload


uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload


```from datetime import datetime
import time

# 1. 서버의 현재 '로컬' 시간 (타임존 정보 없음)
print("Local Now:", datetime.now())

# 2. 서버가 인식하고 있는 타임존 이름
print("Timezone Name:", time.tzname)

# 3. 현재 시스템 타임존의 상세 정보 확인
print("System Timezone:", datetime.now().astimezone().tzname())```

uvicorn app.main:app --host 127.0.0.1 --port 7001 --reload --log-level critical

uvicorn app.main:app --host 127.0.0.1 --port 7001 --no-access-log