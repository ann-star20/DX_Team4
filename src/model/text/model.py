from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    ElectraForSequenceClassification,
)

from .config import TextTrainConfig


def _align_seq_cls_heads_with_backbone(model: ElectraForSequenceClassification) -> None:
    """Electra 백본만 device_map=auto 로 GPU에 있을 때 classifier 등을 같은 디바이스로."""
    try:
        dev = next(model.electra.parameters()).device
    except StopIteration:
        return
    for name, child in model.named_children():
        if name != "electra":
            child.to(dev)


def build_model(cfg: TextTrainConfig):
    """KcELECTRA + 7-class 분류 헤드를 로드한다.

    use_qlora=True (기본):
        4-bit NF4 양자화로 베이스 모델 뼈대를 동결하고,
        학습 가능한 LoRA 어댑터만 attention 레이어에 주입한다.
        → VRAM을 크게 줄이면서 분류 성능을 유지.

    use_qlora=False:
        전체 파라미터를 학습하는 풀 파인튜닝.
    """
    if cfg.use_qlora:
        return _build_qlora(cfg)
    return _build_full(cfg)


def _build_full(cfg: TextTrainConfig):
    """풀 파인튜닝 모드."""
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.pretrained_model,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
    )
    _print_params(model)
    return model


def _build_qlora(cfg: TextTrainConfig):
    """4-bit NF4 양자화 + LoRA 어댑터 주입."""
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as e:
        raise ImportError(
            "QLoRA(peft)는 최신 transformers·accelerate 와 함께 써야 합니다.\n"
            "  pip install -U 'transformers>=4.46' 'accelerate>=0.34' peft bitsandbytes\n"
            "  (transformers 4.40 + 최신 peft → EncoderDecoderCache 없음 / 구 accelerate → clear_device_cache 없음)\n"
            f"원인: {e}"
        ) from e

    # ── 1. 4-bit NF4 양자화 설정 ─────────────────────────────────────────
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NF4: 정보 손실 최소화 양자화
        bnb_4bit_use_double_quant=True,      # 이중 양자화로 추가 메모리 절감
        bnb_4bit_compute_dtype=torch.float16,
    )

    # ── 2. 베이스 로드 (4-bit Electra 백본 + 새 분류 헤드) ───────────────
    # HF 이슈: AutoModelForSequenceClassification + 4bit + ignore_mismatched_sizes
    # → missing key 초기화 시 `weight is not an nn.Module` (transformers 4.46+).
    # 해결: 양자화는 ElectraModel(백본)만 로드 후 ElectraForSequenceClassification 으로
    # num_labels 헤드를 새로 붙임 (체크포인트 분류기 가중치는 사용하지 않음).
    config = AutoConfig.from_pretrained(cfg.pretrained_model)
    config.num_labels = cfg.num_labels
    backbone = AutoModel.from_pretrained(
        cfg.pretrained_model,
        quantization_config=bnb_cfg,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model = ElectraForSequenceClassification(config)
    model.electra = backbone
    _align_seq_cls_heads_with_backbone(model)

    # ── 3. kbit 학습을 위한 전처리 (그라디언트 체크포인트 등) ───────────
    model = prepare_model_for_kbit_training(model)

    # ── 4. LoRA 어댑터 주입 ───────────────────────────────────────────────
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias="none",
        inference_mode=False,
        # Electra: 새로 붙인 분류 헤드는 LoRA 대상이 아니므로 학습 시 함께 저장해야 함
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora_cfg)

    _print_params(model)
    return model


def _print_params(model) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100 * trainable / total if total > 0 else 0
    logger.debug(
        f"Parameters — total: {total:,}  trainable: {trainable:,}  ({pct:.2f}%)"
    )


def load_model_for_inference(
    cfg: TextTrainConfig,
    ckpt_dir: Path | str,
    device: torch.device,
):
    """학습 산출물(`best/` 등)에서 추론용 모델을 로드한다.

    - `.pt` 파일이면 state_dict를 로드하여 모델에 적용.
    - `adapter_config.json` 이 있으면 QLoRA 어댑터로 간주: 4-bit 베이스 + PEFT 로드.
    - 없으면 풀 가중치 체크포인트로 `from_pretrained` 후 `device`로 이동.
    """
    ckpt_dir = Path(ckpt_dir).expanduser().resolve()

    if ckpt_dir.is_file() and ckpt_dir.suffix == ".pt":
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.pretrained_model,
            num_labels=cfg.num_labels,
            ignore_mismatched_sizes=True,
        )

        checkpoint = torch.load(str(ckpt_dir), map_location=device)
        if hasattr(checkpoint, "state_dict"):
            checkpoint = checkpoint.state_dict()
        elif isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]

        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint, strict=False)
        elif isinstance(checkpoint, torch.nn.Module):
            model = checkpoint
        else:
            raise RuntimeError(
                f"알 수 없는 .pt 체크포인트 형식입니다: {ckpt_dir}"
            )

        model.eval()
        return model.to(device)

    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"체크포인트 폴더가 없습니다: {ckpt_dir}")

    adapter_cfg = ckpt_dir / "adapter_config.json"

    if adapter_cfg.exists():
        if device.type != "cuda":
            raise RuntimeError(
                "QLoRA 체크포인트 평가는 CUDA GPU 환경에서만 지원됩니다 "
                "(bitsandbytes 4-bit 로드)."
            )
        try:
            from peft import PeftModel, prepare_model_for_kbit_training
        except ImportError as e:
            raise ImportError(
                "QLoRA 체크포인트 로드에 peft가 필요합니다. pip install peft bitsandbytes"
            ) from e

        with open(adapter_cfg, encoding="utf-8") as f:
            meta = json.load(f)
        base_id = meta.get("base_model_name_or_path") or cfg.pretrained_model

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        bcfg = AutoConfig.from_pretrained(base_id)
        bcfg.num_labels = cfg.num_labels
        backbone = AutoModel.from_pretrained(
            base_id,
            quantization_config=bnb_cfg,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        base = ElectraForSequenceClassification(bcfg)
        base.electra = backbone
        _align_seq_cls_heads_with_backbone(base)
        base = prepare_model_for_kbit_training(base)
        model = PeftModel.from_pretrained(base, str(ckpt_dir))
        model.eval()
        return model

    model = AutoModelForSequenceClassification.from_pretrained(
        str(ckpt_dir),
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
    )
    model.eval()
    return model.to(device)
