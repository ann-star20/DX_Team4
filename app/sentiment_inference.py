from __future__ import annotations

import io
import sys
import threading
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf

# app/ 내부에 위치하므로 프로젝트 루트는 한 단계 위
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


def _format_result_from_probs(probs: np.ndarray | Any, labels: list[str]) -> dict[str, float | str]:
    try:
        import torch
        is_torch_tensor = isinstance(probs, torch.Tensor)
    except Exception:
        is_torch_tensor = False

    if is_torch_tensor:
        probs_np = probs.detach().cpu().numpy().astype(float)
    else:
        probs_np = np.asarray(probs, dtype=float)

    top_idx = int(np.argmax(probs_np))
    result: dict[str, float | str] = {
        "top_emotion": labels[top_idx],
        "top_prob": float(probs_np[top_idx]),
    }
    # 확률 기반 기대값 계산(세션 불일치 지수)에서 재사용할 수 있도록 전체 분포를 함께 반환
    result["probs"] = {
        label: float(prob)
        for label, prob in zip(labels, probs_np.tolist())
    }
    if top_idx != _NEUTRAL_IDX:
        result["neutral_prob"] = float(probs_np[_NEUTRAL_IDX])
    return result


def _load_audio_runtime() -> dict[str, Any]:
    global _audio_runtime
    if _audio_runtime is None:
        with _runtime_lock:
            if _audio_runtime is None:
                import torch
                from peft import LoraConfig, TaskType, get_peft_model
                from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor

                if not _AUDIO_CKPT.is_file():
                    raise FileNotFoundError(f"Audio checkpoint not found: {_AUDIO_CKPT}")

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
                    "facebook/hubert-base-ls960",
                    sampling_rate=TARGET_SR,
                    return_attention_mask=True,
                )
                _audio_runtime = {"device": device, "model": model, "fe": fe}
    return _audio_runtime


def _find_text_checkpoint() -> tuple[Path, str]:
    if _TEXT_PT_CKPT.is_file():
        return _TEXT_PT_CKPT, "pt"

    for sub in ("best", "last"):
        d = _TEXT_DIR / sub
        if d.is_dir() and any((d / f).exists() for f in ("config.json", "adapter_config.json")):
            return d, "hf_dir"

    if _TEXT_DIR.is_dir():
        for child in sorted(_TEXT_DIR.iterdir()):
            if child.is_dir() and any((child / f).exists() for f in ("config.json", "adapter_config.json")):
                return child, "hf_dir"

    raise FileNotFoundError(f"Text checkpoint not found under: {_TEXT_DIR}")


def _load_peft_from_pt(pt_path: Path, cfg: TextTrainConfig, device):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification

    state = torch.load(pt_path, map_location="cpu", weights_only=False)
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg.pretrained_model,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
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


def _tokenizer_source(ckpt_dir: Path | None, pretrained_fallback: str) -> str:
    if ckpt_dir is None or not ckpt_dir.is_dir():
        return pretrained_fallback
    for name in ("tokenizer_config.json", "tokenizer.json", "vocab.txt"):
        if (ckpt_dir / name).exists():
            return str(ckpt_dir)
    return pretrained_fallback


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

                tok_src = _tokenizer_source(ckpt_path if kind == "hf_dir" else None, cfg.pretrained_model)
                tokenizer = AutoTokenizer.from_pretrained(tok_src)

                if kind == "pt":
                    model = _load_peft_from_pt(ckpt_path, cfg, device)
                else:
                    model = load_model_for_inference(cfg, ckpt_path, device)
                    model.eval()

                _text_runtime = {
                    "device": device,
                    "cfg": cfg,
                    "tokenizer": tokenizer,
                    "model": model,
                }
    return _text_runtime


def _load_soft_voting_params() -> dict[str, float]:
    global _soft_voting_params
    if _soft_voting_params is None:
        with _runtime_lock:
            if _soft_voting_params is None:
                if not _SOFT_VOTING_CSV.is_file():
                    raise FileNotFoundError(f"Soft voting config not found: {_SOFT_VOTING_CSV}")
                raw = np.genfromtxt(
                    str(_SOFT_VOTING_CSV),
                    delimiter=",",
                    names=True,
                    dtype=None,
                    encoding="utf-8",
                )
                row = raw[0] if getattr(raw, "shape", ()) else raw
                _soft_voting_params = {
                    "w_text": float(row["w_text_star"]),
                    "w_audio": float(row["w_audio_star"]),
                    "neutral_margin": float(row["neutral_margin_star"]),
                }
    return _soft_voting_params


def _to_mono_resampled_float32(data: np.ndarray, sr: int) -> np.ndarray:
    audio = np.asarray(data, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1) if audio.shape[0] > audio.shape[1] else audio.mean(axis=0)
    audio = audio.reshape(-1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
    return audio.astype(np.float32)


def _find_ffmpeg() -> str:
    """ffmpeg 실행 경로를 찾는다. 없으면 RuntimeError."""
    import shutil

    path = shutil.which("ffmpeg")
    if path:
        return path
    for candidate in (
        Path(sys.prefix) / "Library" / "bin" / "ffmpeg.exe",
        Path(sys.prefix) / "Scripts" / "ffmpeg.exe",
        Path(r"C:\Users\user\anaconda3\Library\bin\ffmpeg.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("ffmpeg를 찾을 수 없습니다. conda install ffmpeg 또는 PATH 설정이 필요합니다.")


def _decode_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """WAV/FLAC/OGG는 soundfile로, WebM/Opus 등은 ffmpeg subprocess로 디코딩."""
    import subprocess

    # 1차: soundfile (WAV, FLAC, OGG 등 libsndfile 지원 포맷)
    try:
        with sf.SoundFile(io.BytesIO(audio_bytes)) as f:
            data = f.read(dtype="float32")
            sr = int(f.samplerate)
        return _to_mono_resampled_float32(data, sr)
    except Exception:
        pass

    # 2차: ffmpeg subprocess (WebM/Opus 등 브라우저 녹음 포맷)
    ffmpeg = _find_ffmpeg()
    proc = subprocess.run(
        [
            ffmpeg, "-y",
            "-i", "pipe:0",
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ar", str(TARGET_SR),
            "-ac", "1",
            "pipe:1",
        ],
        input=audio_bytes,
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace")[:300]
        raise RuntimeError(f"ffmpeg decode failed (rc={proc.returncode}): {stderr}")
    if not proc.stdout:
        raise RuntimeError("ffmpeg produced no output")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def _preprocess_audio_array(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        raise ValueError("Invalid audio input: empty waveform")
    if AUDIO_USE_PEAK_NORMALIZE:
        wav = peak_normalize(wav)
    wav = trim_silence(wav)
    if not is_valid_duration(wav):
        raise ValueError("Invalid audio input: duration out of supported range")
    return wav


def _predict_audio_probs(audio_bytes_or_array) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    runtime = _load_audio_runtime()
    device = runtime["device"]
    model = runtime["model"]
    fe = runtime["fe"]

    if isinstance(audio_bytes_or_array, (bytes, bytearray)):
        wav = _decode_audio_bytes(bytes(audio_bytes_or_array))
    else:
        wav = np.asarray(audio_bytes_or_array, dtype=np.float32)
        if wav.ndim != 1:
            wav = wav.reshape(-1)
    wav = _preprocess_audio_array(wav)

    max_len = int(MAX_DURATION_SEC * TARGET_SR)
    enc = fe(
        wav,
        sampling_rate=TARGET_SR,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_values = enc["input_values"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.inference_mode():
        logits = model(input_values=input_values, attention_mask=attention_mask).logits
        probs = F.softmax(logits, dim=-1)[0]
    return probs.detach().cpu().numpy().astype(float)


def _predict_text_probs(text_string: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    if not isinstance(text_string, str) or not text_string.strip():
        raise ValueError("text_string must be a non-empty string")

    runtime = _load_text_runtime()
    device = runtime["device"]
    cfg = runtime["cfg"]
    tokenizer = runtime["tokenizer"]
    model = runtime["model"]

    enc = tokenizer(
        text_string,
        truncation=True,
        max_length=cfg.max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1)[0]
    return probs.detach().cpu().numpy().astype(float)


def analyze_audio_sentiment(audio_bytes_or_array) -> dict:
    probs = _predict_audio_probs(audio_bytes_or_array)
    return _format_result_from_probs(probs, _EMOTION_LABELS_KO)


def analyze_text_sentiment(text_string) -> dict:
    probs = _predict_text_probs(text_string)
    return _format_result_from_probs(probs, EMOTION_NAMES)


def apply_soft_voting(audio_bytes_or_array, text_string) -> dict:
    """audio/text 확률을 soft_voting.csv 가중치로 결합해 최종 감정을 반환."""
    audio_probs = _predict_audio_probs(audio_bytes_or_array)
    text_probs = _predict_text_probs(text_string)
    params = _load_soft_voting_params()

    combined = (params["w_text"] * text_probs) + (params["w_audio"] * audio_probs)
    top_idx = int(np.argmax(combined))

    # 모델 1위 감정이 다르면 불일치도 증가, 같으면 0.
    text_top = int(np.argmax(text_probs))
    audio_top = int(np.argmax(audio_probs))
    incongruence_score = 0.0
    if text_top != audio_top:
        incongruence_score = float(
            abs(text_probs[text_top] - audio_probs[text_top])
            + abs(audio_probs[audio_top] - text_probs[audio_top])
        )

    # 중립 우세 차이가 margin 미만이면 비중립 최댓값을 우선.
    non_neutral_indices = [i for i in range(len(combined)) if i != _NEUTRAL_IDX]
    best_non_neutral_idx = int(max(non_neutral_indices, key=lambda i: combined[i]))
    if top_idx == _NEUTRAL_IDX:
        if float(combined[_NEUTRAL_IDX] - combined[best_non_neutral_idx]) < params["neutral_margin"]:
            top_idx = best_non_neutral_idx

    result = _format_result_from_probs(combined, _EMOTION_LABELS_KO)
    result["top_emotion"] = _EMOTION_LABELS_KO[top_idx]
    result["top_prob"] = float(combined[top_idx])
    if top_idx != _NEUTRAL_IDX:
        result["neutral_prob"] = float(combined[_NEUTRAL_IDX])
    result["incongruence_score"] = incongruence_score
    return result


def apply_soft_voting_text_only(text_string: str) -> dict:
    """오디오가 없을 때 음성 분기는 중립(1-hot)으로 두고 soft voting 결합한다."""
    if not isinstance(text_string, str) or not text_string.strip():
        raise ValueError("text_string must be a non-empty string")

    text_probs = _predict_text_probs(text_string)
    params = _load_soft_voting_params()
    n = len(text_probs)
    audio_probs = np.zeros(n, dtype=float)
    audio_probs[_NEUTRAL_IDX] = 1.0

    combined = (params["w_text"] * text_probs) + (params["w_audio"] * audio_probs)
    top_idx = int(np.argmax(combined))

    text_top = int(np.argmax(text_probs))
    audio_top = int(np.argmax(audio_probs))
    incongruence_score = 0.0
    if text_top != audio_top:
        incongruence_score = float(
            abs(text_probs[text_top] - audio_probs[text_top])
            + abs(audio_probs[audio_top] - text_probs[audio_top])
        )

    non_neutral_indices = [i for i in range(len(combined)) if i != _NEUTRAL_IDX]
    best_non_neutral_idx = int(max(non_neutral_indices, key=lambda i: combined[i]))
    if top_idx == _NEUTRAL_IDX:
        if float(combined[_NEUTRAL_IDX] - combined[best_non_neutral_idx]) < params["neutral_margin"]:
            top_idx = best_non_neutral_idx

    result = _format_result_from_probs(combined, _EMOTION_LABELS_KO)
    result["top_emotion"] = _EMOTION_LABELS_KO[top_idx]
    result["top_prob"] = float(combined[top_idx])
    if top_idx != _NEUTRAL_IDX:
        result["neutral_prob"] = float(combined[_NEUTRAL_IDX])
    result["incongruence_score"] = incongruence_score
    return result

