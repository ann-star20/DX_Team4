"""프로젝트 경로 및 전역 상수 설정.

전처리 동작은 docs/architecture_guide.md 와 정합한다.
  - 텍스트 Part 1: 라벨 Part 0, 정제·KEMDy 태그는 text_cleaner
  - 음성 Part 2: HuBERT용 16kHz mono 파형 WAV (Mel/HDF5는 레거시 CNN 경로)
"""
from pathlib import Path

# ── 루트 경로 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT_ROOT / "data" / "Raw"
PREPROCESSED_ROOT = PROJECT_ROOT / "data" / "PreProcessed"

# ── 출력 경로 ─────────────────────────────────────────────────────────────────
TEXT_OUT = PREPROCESSED_ROOT / "text"
AUDIO_OUT = PREPROCESSED_ROOT / "audio"

# ── 원시 데이터 경로 ───────────────────────────────────────────────────────────
KEMDY20_ROOT = RAW_ROOT / "KEMDy20_v1_2"
KEMDY20_ANNO_DIR = KEMDY20_ROOT / "annotation"
KEMDY20_WAV_DIR = KEMDY20_ROOT / "wav"

EMOTION_CLF_ROOT = RAW_ROOT / "감정 분류를 위한 대화 음성 데이터셋"
EMOTION_CLF_YEARS = ["4차년도", "5차년도", "5차년도_2차"]

# emotion_clf 공유 split 파일 경로
# 음성 전처리가 먼저 생성 → 텍스트 전처리가 읽어서 동일한 split 사용
# 형식: sample_id(str) → split(train/val/test) 매핑 CSV
EMOTION_CLF_SPLIT_FILE = PREPROCESSED_ROOT / "emotion_clf_split_keys.csv"

# ── 오디오 전처리 파라미터 (architecture_guide Part 2 · HuBERT 입력 정합) ───────
TARGET_SR = 16_000          # 16kHz — LS960 계열 HuBERT와 동일
MIN_DURATION_SEC = 0.5      # 최소 길이 필터 (초)
MAX_DURATION_SEC = 30.0     # 최대 길이 필터 (초); 학습 시 max_length(샘플) ≒ 이 값 × TARGET_SR
TOP_DB = 30                 # librosa.effects.trim — 침묵 트림 (가이드 TOP_DB=30)
# True: 피크 정규화 적용. False: 가이드에 따른 피크 정규화 ablation(HF FeatureExtractor만 사용)
AUDIO_USE_PEAK_NORMALIZE = True

# ── 소스 가중치 (Part Pre 5) source_weight) ───────────────────────────────────
SOURCE_WEIGHTS: dict[str, float] = {
    "kemdy20": 3.0,
    "emotion_clf": 1.0,
}

# ── 랜덤 시드 ─────────────────────────────────────────────────────────────────
SEED = 42
