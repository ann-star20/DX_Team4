from __future__ import annotations

import csv
import io
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model.text.config import EMOTION_NAMES, TextTrainConfig
from preprocess.audio.audio_utils import is_valid_duration, peak_normalize, trim_silence
from preprocess.config import AUDIO_USE_PEAK_NORMALIZE, MAX_DURATION_SEC, TARGET_SR

_EMOTION_LABELS_KO = ["기쁨", "슬픔", "분노", "공포", "놀람", "중립", "혐오"]
_NEUTRAL_IDX = _EMOTION_LABELS_KO.index("중립")

_AUDIO_CKPT = PROJECT_ROOT / "models" / "audio" / "best_hubert_lora.pt"
_TEXT_PT_CKPT = PROJECT_ROOT / "models" / "text" / "best_kcelectra_lora.pt"
_TEXT_DIR = PROJECT_ROOT / "models" / "text"
_SOFT_VOTING_CSV = PROJECT_ROOT / "models" / "soft_voting.csv"

_audio_runtime: dict[str, Any] | None = None
_text_runtime: dict[str, Any] | None = None
_soft_voting_params: dict[str, float] | None = None
_runtime_lock = threading.Lock()


def _get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _format_result_from_probs(probs: np.ndarray | Any, labels: list[str]) -> dict[str, Any]:
    probs_np = np.asarray(probs, dtype=float)
    top_idx = int(np.argmax(probs_np))
    out: dict[str, Any] = {
        "top_emotion": labels[top_idx],
        "top_prob": float(probs_np[top_idx]),
        "probs": {labels[i]: float(probs_np[i]) for i in range(len(labels))},
    }
    if top_idx != _NEUTRAL_IDX:
        out["neutral_prob"] = float(probs_np[_NEUTRAL_IDX])
    return out


def _find_text_checkpoint() -> tuple[Path, str]:
    if _TEXT_PT_CKPT.is_file():
        return _TEXT_PT_CKPT, "pt"
    for sub in ("best", "last"):
        d = _TEXT_DIR / sub
        if d.is_dir() and any((d / f).exists() for f in ("config.json", "adapter_config.json")):
            return d, "hf_dir"
    raise FileNotFoundError(f"Text checkpoint not found under: {_TEXT_DIR}")


def _load_peft_from_pt(pt_path: Path, cfg: TextTrainConfig, device):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    state = torch.load(pt_path, map_location="cpu", weights_only=False)
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg.pretrained_model, num_labels=cfg.num_labels, ignore_mismatched_sizes=True
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=cfg.lora_dropout,
        target_modules=["query", "key", "value", "dense"],
        bias="none",
        modules_to_save=["classifier"],
    )
    model = get_peft_model(base, lora_cfg)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


def _load_audio_runtime() -> dict[str, Any]:
    global _audio_runtime
    if _audio_runtime is None:
        with _runtime_lock:
            if _audio_runtime is None:
                import torch
                from peft import LoraConfig, TaskType, get_peft_model
                from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

                device = _get_device()
                state = torch.load(str(_AUDIO_CKPT), map_location="cpu", weights_only=False)
                base = HubertForSequenceClassification.from_pretrained(
                    "facebook/hubert-base-ls960",
                    num_labels=len(_EMOTION_LABELS_KO),
                    ignore_mismatched_sizes=True,
                )
                lora_cfg = LoraConfig(
                    task_type=TaskType.SEQ_CLS,
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                    bias="none",
                    modules_to_save=["classifier", "projector"],
                )
                model = get_peft_model(base, lora_cfg)
                model.load_state_dict(state, strict=True)
                model.eval()
                model.to(device)
                fe = Wav2Vec2FeatureExtractor.from_pretrained(
                    "facebook/hubert-base-ls960", sampling_rate=TARGET_SR, return_attention_mask=True
                )
                _audio_runtime = {"device": device, "model": model, "fe": fe}
    return _audio_runtime


def _load_text_runtime() -> dict[str, Any]:
    global _text_runtime
    if _text_runtime is None:
        with _runtime_lock:
            if _text_runtime is None:
                from transformers import AutoTokenizer
                from model.text.model import load_model_for_inference

                cfg = TextTrainConfig()
                ckpt_path, kind = _find_text_checkpoint()
                device = _get_device()
                tokenizer = AutoTokenizer.from_pretrained(cfg.pretrained_model)
                if kind == "pt":
                    model = _load_peft_from_pt(ckpt_path, cfg, device)
                else:
                    model = load_model_for_inference(cfg, ckpt_path, device)
                    model.eval()
                _text_runtime = {"device": device, "cfg": cfg, "tokenizer": tokenizer, "model": model}
    return _text_runtime


def load_soft_voting_params() -> dict[str, float]:
    global _soft_voting_params
    if _soft_voting_params is None:
        with _runtime_lock:
            if _soft_voting_params is None:
                with _SOFT_VOTING_CSV.open("r", encoding="utf-8") as f:
                    row = next(csv.DictReader(f))
                _soft_voting_params = {
                    "w_text": float(row["w_text_star"]),
                    "w_audio": float(row["w_audio_star"]),
                    "neutral_margin": float(row["neutral_margin_star"]),
                }
    return _soft_voting_params


def _decode_via_ffmpeg(audio_bytes: bytes) -> np.ndarray:
    """WebM/Opus·MP4 등 libsndfile 미지원 포맷 → ffmpeg로 mono float32 @ TARGET_SR."""
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError(
            "ffmpeg가 PATH에 없습니다. 브라우저 WebM/Opus 디코딩을 위해 ffmpeg를 설치해 주세요."
        )
    proc = subprocess.run(
        [
            exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(TARGET_SR),
            "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
        timeout=120,
    )
    if proc.returncode != 0:
        err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(err or f"ffmpeg exit {proc.returncode}")
    raw = proc.stdout
    if not raw:
        raise RuntimeError("ffmpeg 출력이 비어 있습니다.")
    if len(raw) % 4 != 0:
        raise RuntimeError("ffmpeg 출력 길이가 float32 정렬이 아닙니다.")
    return np.frombuffer(raw, dtype=np.float32).reshape(-1).copy()


def _decode_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    if not audio_bytes:
        raise ValueError("empty audio")

    # 1) WAV/FLAC 등 libsndfile이 읽는 포맷
    try:
        with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
            data = f.read(dtype="float32")
            sr = int(f.samplerate)
        audio = np.asarray(data, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1) if audio.shape[0] > audio.shape[1] else audio.mean(axis=0)
        audio = audio.reshape(-1)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
        return audio
    except Exception:
        pass

    # 2) librosa + audioread(환경에 따라 ffmpeg 사용) — 일부 컨테이너
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
        audio = np.asarray(y, dtype=np.float32).reshape(-1)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
        return audio
    except Exception:
        pass

    # 3) 브라우저 MediaRecorder WebM/Opus 등
    return _decode_via_ffmpeg(audio_bytes)


def _preprocess_audio_array(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if AUDIO_USE_PEAK_NORMALIZE:
        wav = peak_normalize(wav)
    wav = trim_silence(wav)
    if not is_valid_duration(wav):
        raise ValueError("Invalid audio duration")
    return wav


def analyze_audio_sentiment(audio_bytes_or_array) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    runtime = _load_audio_runtime()
    device = runtime["device"]
    model = runtime["model"]
    fe = runtime["fe"]
    if isinstance(audio_bytes_or_array, (bytes, bytearray)):
        wav = _decode_audio_bytes(bytes(audio_bytes_or_array))
    else:
        wav = np.asarray(audio_bytes_or_array, dtype=np.float32).reshape(-1)
    wav = _preprocess_audio_array(wav)

    enc = fe(
        wav,
        sampling_rate=TARGET_SR,
        max_length=int(MAX_DURATION_SEC * TARGET_SR),
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.inference_mode():
        logits = model(
            input_values=enc["input_values"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits
        probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy().astype(float)
    return _format_result_from_probs(probs, _EMOTION_LABELS_KO)


def analyze_text_sentiment(text_string: str) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    runtime = _load_text_runtime()
    device = runtime["device"]
    cfg = runtime["cfg"]
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]

    enc = tokenizer(text_string, truncation=True, max_length=cfg.max_length, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1)[0].detach().cpu().numpy().astype(float)
    return _format_result_from_probs(probs, EMOTION_NAMES)

