"""오디오 공통 전처리 유틸리티 (docs/architecture_guide.md Part 2).

HuBERT 학습·추론 경로:
  - 디스크: 16kHz mono WAV (내부 연산은 float32, 저장은 PCM_16 — 표준 음성 WAV).
  - 배치 단계에서 transformers Wav2Vec2FeatureExtractor로 input_values·attention_mask 생성.

파이프라인 (Mel+CNN 레거시와 공통 저장 형식만 공유, 멜 스펙·HDF5는 pack_mel_hdf5):
  load (슬라이스) → mono → 16kHz 리샘플 → (선택) 피크 정규화 → 침묵 트림 → 길이 검증
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

from preprocess.config import (
    AUDIO_USE_PEAK_NORMALIZE,
    MAX_DURATION_SEC,
    MIN_DURATION_SEC,
    TARGET_SR,
    TOP_DB,
)


def load_and_slice(
    wav_path: Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> np.ndarray | None:
    """WAV 파일을 로드 (필요 시 시간 구간 슬라이스).

    Returns
    -------
    float32 mono 1-D ndarray (TARGET_SR) 또는 None (로드 실패 시)
    """
    try:
        if start_sec is not None and end_sec is not None:
            # 긴 파일은 soundfile 부분 로드 시도 (빠름)
            # ADPCM 등 seek 불가 포맷은 SoundFileError 발생 → librosa fallback
            try:
                with sf.SoundFile(wav_path) as f:
                    orig_sr = f.samplerate
                    start_frame = int(start_sec * orig_sr)
                    end_frame = int(end_sec * orig_sr)
                    f.seek(start_frame)
                    frames = f.read(end_frame - start_frame, dtype="float32")
            except (sf.SoundFileError, RuntimeError):
                # seek 불가 포맷 → 전체 로드 후 배열 슬라이싱
                y, orig_sr = librosa.load(str(wav_path), sr=None, mono=False)
                # librosa mono=False: stereo=(channels, samples), mono=(samples,)
                if y.ndim == 2:
                    y = y.T  # (samples, channels)
                s = int(start_sec * orig_sr)
                e = int(end_sec * orig_sr)
                frames = y[s:e].astype(np.float32)
        else:
            frames, orig_sr = sf.read(wav_path, dtype="float32")

        # mono 변환
        if frames.ndim == 2:
            frames = frames.mean(axis=1)

        # 리샘플
        if orig_sr != TARGET_SR:
            frames = librosa.resample(frames, orig_sr=orig_sr, target_sr=TARGET_SR)

        return frames.astype(np.float32)

    except Exception as e:
        logger.warning("WAV 로드 실패 [%s]: %s — %s", wav_path, type(e).__name__, e)
        return None


def peak_normalize(audio: np.ndarray) -> np.ndarray:
    """피크 기준 정규화 (클리핑 방지)."""
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95
    return audio


def trim_silence(audio: np.ndarray) -> np.ndarray:
    """무음 구간 제거.

    librosa.effects.trim 은 내부적으로 의존성 체인을 더 타서
    일부 Windows 환경에서 _lzma import 에러를 유발할 수 있다.
    여기서는 진폭 기반 dB 임계값으로 앞/뒤 무음을 제거한다.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x

    peak = float(np.max(np.abs(x)))
    if peak <= 0.0:
        return x

    # peak 대비 상대 dB 마스크 (TOP_DB=60이면 -60dB보다 큰 구간 유지)
    eps = 1e-12
    rel_db = 20.0 * np.log10((np.abs(x) + eps) / (peak + eps))
    keep = rel_db > (-float(TOP_DB))
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return x
    return x[int(idx[0]) : int(idx[-1]) + 1]


def is_valid_duration(audio: np.ndarray) -> bool:
    """길이 이상치 필터 (MIN ~ MAX 초)."""
    dur = len(audio) / TARGET_SR
    return MIN_DURATION_SEC <= dur <= MAX_DURATION_SEC


def file_hash(wav_path: Path) -> str:
    """WAV 파일 MD5 해시 (중복 제거용)."""
    h = hashlib.md5()
    with open(wav_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def save_wav(audio: np.ndarray, out_path: Path) -> None:
    """float32 mono 파형을 표준 PCM_16 WAV로 저장 (로더가 float으로 복원해 HuBERT에 공급)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, TARGET_SR, subtype="PCM_16")


def preprocess_audio_with_reason(
    wav_path: Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> tuple[np.ndarray | None, str | None]:
    """load → mono → resample → peak_normalize → trim_silence → duration 검증.

    성공 시 (audio, None). 실패 시 (None, 이유코드):
      ``load_failed`` — 디코드/슬라이스 실패 또는 빈 배열
      ``duration``   — 트림 후 MIN~MAX 초 밖 (짧은 발화·거의 무음 등)
    """
    audio = load_and_slice(wav_path, start_sec, end_sec)
    if audio is None or audio.size == 0:
        return None, "load_failed"
    if AUDIO_USE_PEAK_NORMALIZE:
        audio = peak_normalize(audio)
    audio = trim_silence(audio)
    if not is_valid_duration(audio):
        return None, "duration"
    return audio, None


def preprocess_audio(
    wav_path: Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> np.ndarray | None:
    """load → mono → resample → peak_normalize → trim_silence → duration 검증.

    Returns None if invalid.
    """
    audio, _ = preprocess_audio_with_reason(wav_path, start_sec, end_sec)
    return audio


def duration_sec(audio: np.ndarray) -> float:
    return len(audio) / TARGET_SR
