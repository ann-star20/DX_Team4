from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# EMOTION_NAMES = ["기쁨", "슬픔/우울", "분노/불쾌", "불안/공포", "놀람/당황", "중립", "혐오"]
EMOTION_NAMES = ["기쁨", "슬픔", "분노", "공포", "놀람", "중립", "혐오"]



@dataclass
class TextTrainConfig:
    # ── 모델 ──────────────────────────────────────────────────────────────
    pretrained_model: str = "beomi/kcelectra-base-v2022"
    max_length: int = 128
    num_labels: int = 7

    # ── QLoRA 설정 ────────────────────────────────────────────────────────
    # use_qlora=True : 4-bit NF4 양자화 + LoRA 어댑터 (VRAM 절감)
    # use_qlora=False: 풀 파인튜닝 (기존 동작)
    use_qlora: bool = True
    lora_r: int = 16               # LoRA 랭크 (클수록 표현력↑, VRAM↑)
    lora_alpha: int = 32           # LoRA 스케일링 계수 (보통 2×lora_r)
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["query", "key", "value"]
    )

    # ── 학습 하이퍼파라미터 ───────────────────────────────────────────────
    epochs: int = 5
    # None = 비활성화. 검증 macro_f1 기준 연속 N 에폭 동안 최고 기록을 갱신하지 않으면 중단
    early_stopping_patience: Optional[int] = None
    batch_size: int = 32
    lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42

    # ── 가중치 실험 ───────────────────────────────────────────────────────
    # class_weight_mode: "none" | "inv" | "sqrt_inv"
    #   none     : CrossEntropyLoss 균등 (베이스라인)
    #   inv      : 클래스 빈도 역수 비례
    #   sqrt_inv : 클래스 빈도 역수 제곱근 (완만한 보정)
    class_weight_mode: str = "none"

    # context_weight: dialog_type=="context" 샘플의 샘플링 가중치
    #   1.0 = 베이스라인 (맥락/단발 동일 확률)
    #   2.0+ = 맥락 데이터를 더 자주 샘플링 (WeightedRandomSampler)
    context_weight: float = 1.0

    # ── 데이터 경로 ───────────────────────────────────────────────────────
    train_pool: Path = Path("data/PreProcessed/text/text_train_pool.parquet")
    val_pool: Path = Path("data/PreProcessed/text/text_val_pool.parquet")

    # 빠른 실험을 위한 샘플 수 제한 (None = 전체 사용)
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = 50_000

    # ── 출력 ─────────────────────────────────────────────────────────────
    output_dir: Path = Path("checkpoints/text")
    log_every_steps: int = 200
