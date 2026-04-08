#!/usr/bin/env python3
"""텍스트 감정 분류 단일 문장 추론 CLI.

프로젝트 루트에서 실행:

    python3 scripts/text_infer_cli.py
    python3 scripts/text_infer_cli.py --text "오늘 기분이 좋아"

체크포인트 탐색 순서 (`models/text/` 기준):
  1. `best_kcelectra_lora.pt` — 노트북 등에서 저장한 PEFT state_dict
  2. `best/` — `src/model/text/train.py` 가 저장한 HF/PEFT 디렉터리
  3. 그 외 `models/text/<하위폴더>` 중 `config.json` 또는 `adapter_config.json` 이 있는 첫 폴더
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# 프로젝트 루트 = scripts/ 의 부모
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

from model.text.config import EMOTION_NAMES, TextTrainConfig
from model.text.model import load_model_for_inference

# 노트북에서 export한 .pt 와 학습 스크립트 산출물이 LoRA rank 가 다를 수 있음
_LORA_R_PT = 8
_LORA_ALPHA_PT = 16


def _find_text_checkpoint(
    root: Path,
    override: Path | None,
) -> tuple[Path, str]:
    """반환: (경로, 'pt' | 'hf_dir')."""
    if override is not None:
        p = override.expanduser().resolve()
        if p.is_file() and p.suffix.lower() == ".pt":
            return p, "pt"
        if p.is_dir():
            return p, "hf_dir"
        raise FileNotFoundError(f"--ckpt 경로 없음: {p}")

    text_dir = root / "models" / "text"
    pt = text_dir / "best_kcelectra_lora.pt"
    if pt.is_file():
        return pt, "pt"

    for sub in ("best", "last"):
        d = text_dir / sub
        if d.is_dir() and any((d / f).exists() for f in ("config.json", "adapter_config.json")):
            return d, "hf_dir"

    if text_dir.is_dir():
        for child in sorted(text_dir.iterdir()):
            if child.is_dir() and any(
                (child / f).exists() for f in ("config.json", "adapter_config.json")
            ):
                return child, "hf_dir"

    raise FileNotFoundError(
        "텍스트 체크포인트를 찾지 못했습니다.\n"
        f"  기대: {text_dir / 'best_kcelectra_lora.pt'}\n"
        f"     또는: {text_dir / 'best'}/ (HF/PEFT 폴더)\n"
        "  --ckpt 로 경로를 직접 지정할 수 있습니다."
    )


def _tokenizer_source(ckpt_dir: Path | None, pretrained_fallback: str) -> str:
    if ckpt_dir is None or not ckpt_dir.is_dir():
        return pretrained_fallback
    for name in ("tokenizer_config.json", "tokenizer.json", "vocab.txt"):
        if (ckpt_dir / name).exists():
            return str(ckpt_dir)
    return pretrained_fallback


def _load_peft_from_pt(pt_path: Path, cfg: TextTrainConfig, device: torch.device):
    """MPS/CPU/CUDA 공통: float PEFT + 전체 state_dict (.pt)."""
    state = torch.load(pt_path, map_location="cpu", weights_only=False)
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg.pretrained_model,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=_LORA_R_PT,
        lora_alpha=_LORA_ALPHA_PT,
        lora_dropout=cfg.lora_dropout,
        target_modules=["query", "key", "value", "dense"],
        bias="none",
        modules_to_save=["classifier"],
    )
    model = get_peft_model(base, lora_cfg)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device)


def load_text_model(device: torch.device, ckpt_override: Path | None = None):
    cfg = TextTrainConfig()
    path, kind = _find_text_checkpoint(_PROJECT_ROOT, ckpt_override)

    hf_dir_for_tok: Path | None = path if kind == "hf_dir" else None
    tok_src = _tokenizer_source(hf_dir_for_tok, cfg.pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(tok_src)

    if kind == "pt":
        model = _load_peft_from_pt(path, cfg, device)
    else:
        model = load_model_for_inference(cfg, path, device)
        model.eval()

    return tokenizer, model, path, kind


def predict(text: str, tokenizer, model, device: torch.device, max_length: int):
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = F.softmax(logits, dim=-1)[0].cpu()

    idx = int(torch.argmax(probs).item())
    prob_dict = {
        f"{i}:{name}": float(probs[i].item())
        for i, name in enumerate(EMOTION_NAMES)
    }
    return idx, EMOTION_NAMES[idx], prob_dict


def main():
    parser = argparse.ArgumentParser(description="KcELECTRA 텍스트 감정 추론")
    parser.add_argument("--ckpt", type=Path, default=None, help=".pt 파일 또는 HF 체크포인트 폴더")
    parser.add_argument("--text", type=str, default=None, help="한 번만 추론하고 종료")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    cfg = TextTrainConfig()
    logger.debug(f"프로젝트 루트: {_PROJECT_ROOT}")
    logger.debug(f"사용 디바이스: {device}")

    tokenizer, model, ckpt_path, ckpt_kind = load_text_model(device, args.ckpt)
    logger.debug(f"체크포인트 ({ckpt_kind}): {ckpt_path}")

    if args.text is not None:
        idx, label, probs = predict(args.text, tokenizer, model, device, cfg.max_length)
        logger.debug(f"\n예측 라벨: {idx} ({label})")
        for k, v in probs.items():
            logger.debug(f"  {k}: {v:.4f}")
        return

    logger.debug("\n텍스트 감정분석 (종료: 빈 줄 또는 Ctrl+C)\n")
    while True:
        try:
            line = input("문장 입력> ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.debug("\n종료.")
            break
        if not line:
            logger.debug("종료.")
            break
        idx, label_ko, prob_dict = predict(line, tokenizer, model, device, cfg.max_length)
        logger.debug(f"\n예측 라벨: {idx} ({label_ko})")
        logger.debug("클래스별 확률:")
        for k, v in prob_dict.items():
            logger.debug(f"  {k:<14} : {v:.4f}")
        logger.debug("-" * 40)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s: %(message)s",
    )
    main()
