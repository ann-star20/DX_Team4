# 모델·전처리 모듈

이 디렉터리는 FastAPI 감정 추론에서 사용하는 로더와 전처리 코드입니다. 전체 실행 절차는 [프로젝트 README](../README.md)를 따릅니다.

- 텍스트: KcELECTRA 계열 7감정 분류.
- 음성: HuBERT-LoRA, 16 kHz mono 전처리.
- 라벨: 기쁨, 슬픔, 분노, 공포, 놀람, 중립, 혐오.
- 런타임 진입점: app/sentiment_inference.py.
- 모델 가중치·학습 데이터·학습 CLI는 공개 배포 범위에 포함하지 않습니다.

이전에 기재됐던 scripts/text_infer_cli.py와 scripts/audio_infer_cli.py는 현재 공개 저장소에 없으므로 실행 명령에서 제외했습니다. API 실행과 모델 파일 배치는 루트 README를 참고하세요.
