#!/usr/bin/env python3
"""음성 감정 분류 단일 WAV 추론 CLI (HuBERT + LoRA).

프로젝트 루트에서 실행:

    python scripts/audio_infer_cli.py --wav path/to/sample.wav
    python scripts/audio_infer_cli.py --mic --seconds 5
    python scripts/audio_infer_cli.py   # 기본: 마이크로 말해서 반복 테스트

체크포인트(기본):
  - models/audio/best_hubert_lora.pt  (노트북 02 스타일: PEFT state_dict)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# 프로젝트 루트 = scripts/ 의 부모
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch
import torch.nn.functional as F

from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
from peft import LoraConfig, TaskType, get_peft_model

from preprocess.config import (
    AUDIO_USE_PEAK_NORMALIZE,
    MAX_DURATION_SEC,
    TARGET_SR,
)
from preprocess.audio.audio_utils import (
    is_valid_duration,
    peak_normalize,
    preprocess_audio_with_reason,
    trim_silence,
)


EMOTION_LABELS_KO = ["기쁨", "슬픔", "분노", "공포", "놀람", "중립", "혐오"]


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_path(p: str) -> Path:
    pp = Path(p).expanduser()
    if pp.is_absolute() and pp.exists():
        return pp
    cand = _PROJECT_ROOT / pp
    return cand


def _preprocess_like_train_from_array(wav: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    """훈련 전처리와 최대한 동일하게: peak_normalize(옵션) → trim_silence → duration 필터."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return None, "load_failed"
    if AUDIO_USE_PEAK_NORMALIZE:
        wav = peak_normalize(wav)
    wav = trim_silence(wav)
    if not is_valid_duration(wav):
        return None, "duration"
    return wav, None


def load_audio_model(
    ckpt_pt: Path,
    device: torch.device,
    base_hubert_id: str = "facebook/hubert-base-ls960",
    num_labels: int = 7,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
):
    """models/audio/best_hubert_lora.pt (PEFT state_dict) 로드."""
    state = torch.load(str(ckpt_pt), map_location="cpu", weights_only=False)

    base = HubertForSequenceClassification.from_pretrained(
        base_hubert_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        bias="none",
        modules_to_save=["classifier", "projector"],
    )
    model = get_peft_model(base, lora_cfg)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


def load_feature_extractor(base_hubert_id: str = "facebook/hubert-base-ls960", sr: int = 16000):
    return Wav2Vec2FeatureExtractor.from_pretrained(
        base_hubert_id,
        sampling_rate=sr,
        return_attention_mask=True,
    )


def predict_wav(
    wav: np.ndarray,
    fe: Wav2Vec2FeatureExtractor,
    model,
    device: torch.device,
    sr: int = TARGET_SR,
    max_sec: float = MAX_DURATION_SEC,
):
    max_len = int(max_sec * sr)
    enc = fe(
        wav,
        sampling_rate=sr,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_values = enc["input_values"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_values=input_values, attention_mask=attention_mask).logits
        probs = F.softmax(logits, dim=-1)[0].detach().cpu()
    pred = int(torch.argmax(probs).item())
    return pred, probs


def main():
    parser = argparse.ArgumentParser(description="음성 감정 분류 추론 (HuBERT+LoRA)")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/audio/best_hubert_lora.pt",
        help="PEFT state_dict .pt 경로 (기본: models/audio/best_hubert_lora.pt)",
    )
    parser.add_argument("--wav", type=str, default=None, help="WAV 파일 경로")
    parser.add_argument(
        "--mic",
        action="store_true",
        help="마이크로 녹음해서 추론 (sounddevice 필요)",
    )
    parser.add_argument(
        "--mic-loop",
        action="store_true",
        help="마이크 녹음을 반복해서 계속 추론 (기본 모드)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="--mic 사용 시 녹음 길이(초)",
    )
    parser.add_argument(
        "--max-sec",
        type=float,
        default=float(MAX_DURATION_SEC),
        help=f"최대 입력 길이(초) (기본: {MAX_DURATION_SEC})",
    )
    args = parser.parse_args()

    device = _get_device()
    ckpt = _resolve_path(args.ckpt).resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"체크포인트(.pt) 없음: {ckpt}")

    logger.debug(f"프로젝트 루트: {_PROJECT_ROOT}")
    logger.debug(f"사용 디바이스: {device}")
    logger.debug(f"체크포인트: {ckpt}")

    fe = load_feature_extractor(sr=TARGET_SR)
    model = load_audio_model(ckpt, device)

    def _record_from_mic(seconds: float, sr: int = 16000) -> np.ndarray:
        try:
            import sounddevice as sd
        except Exception as e:
            raise RuntimeError(
                "마이크 녹음에는 sounddevice가 필요합니다.\n"
                "설치: pip install sounddevice\n"
                f"(원인: {e})"
            ) from e

        seconds = float(min(seconds, args.max_sec))
        n = int(seconds * sr)
        logger.debug("\n3초 뒤 녹음을 시작합니다.")
        for i in (3, 2, 1):
            logger.debug(f"  {i}...")
            try:
                import time

                time.sleep(1)
            except Exception:
                pass
        logger.debug(f"녹음 시작: {seconds:.1f}s  (sr={sr})")
        rec = sd.rec(frames=n, samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        wav = np.asarray(rec).reshape(-1).astype(np.float32)
        logger.debug("녹음 완료")
        return wav

    def _run_one(wav_path: str):
        p = _resolve_path(wav_path).resolve()
        if not p.exists():
            logger.error(f"파일 없음: {p}")
            return

        # 훈련 전처리와 동일: load→mono→resample→(peak_norm)→trim→duration_filter
        wav, reason = preprocess_audio_with_reason(p)
        if wav is None:
            logger.error(f"전처리 실패({reason}): {p}")
            return

        pred, probs = predict_wav(wav, fe, model, device, sr=TARGET_SR, max_sec=args.max_sec)
        logger.debug(f"\n파일: {p}")
        logger.debug(f"예측 라벨: {pred} ({EMOTION_LABELS_KO[pred]})")
        logger.debug("클래스별 확률:")
        for i, name in enumerate(EMOTION_LABELS_KO):
            logger.debug(f"  {i}:{name:<4} : {float(probs[i]):.4f}")
        logger.debug("-" * 40)

    def _run_mic_once():
        raw = _record_from_mic(args.seconds, sr=TARGET_SR)
        wav, reason = _preprocess_like_train_from_array(raw)
        if wav is None:
            logger.error(
                f"\n전처리 실패({reason}): 마이크 입력 (무음/너무 짧음/너무 김)"
            )
            return
        pred, probs = predict_wav(wav, fe, model, device, sr=TARGET_SR, max_sec=args.max_sec)
        logger.debug(f"\n입력: mic ({args.seconds:.1f}s)")
        logger.debug(f"예측 라벨: {pred} ({EMOTION_LABELS_KO[pred]})")
        logger.debug("클래스별 확률:")
        for i, name in enumerate(EMOTION_LABELS_KO):
            logger.debug(f"  {i}:{name:<4} : {float(probs[i]):.4f}")
        logger.debug("-" * 40)

    def _run_mic_loop():
        logger.debug("\n마이크 반복 테스트 모드 (종료: Enter 또는 Ctrl+C)\n")
        while True:
            try:
                cmd = input("녹음 시작하려면 Enter (종료: q)> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                logger.debug("\n종료.")
                break
            if cmd in ("q", "quit", "exit") or cmd == "":
                # Enter는 '한 번 실행'이 아니라 사용자에게 선택권을 주기 위해 여기서 종료로 처리
                # (실수로 계속 녹음되는 걸 방지)
                if cmd == "":
                    logger.debug("종료.")
                else:
                    logger.debug("종료.")
                break
            if cmd in ("r", "rec", "record", "go", "start"):
                _run_mic_once()
            else:
                logger.debug("명령: r=녹음, q=종료")

    # 기본 동작: 아무 인자 없으면 mic-loop
    if args.mic_loop or (args.wav is None and not args.mic):
        _run_mic_loop()
        return

    if args.mic:
        _run_mic_once()
        return

    if args.wav is not None:
        _run_one(args.wav)
        return
    # (여기 도달하면 인자 구성이 애매한 경우)
    parser.print_help()
    return


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s: %(message)s",
    )
    main()

