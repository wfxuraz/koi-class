from dataclasses import dataclass, fields
from pathlib import Path
from typing import Union

import yaml

_DEFAULTS = {
    "data_dir": "data",
    "backbone": "efficientnetv2_rw_s",
    "image_size": 224,
    "batch_size": 32,
    "epochs": 20,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.1,
    "val_split": 0.2,
    "seed": 42,
    "num_workers": 4,
    "patience": 5,
    "cb_beta": 0.999,
    "output_dir": "runs",
    "top_k": 0,
    "min_prob": 0.0,
}


@dataclass
class Config:
    data_dir: str
    backbone: str
    image_size: int
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    label_smoothing: float
    val_split: float
    seed: int
    num_workers: int
    patience: int
    cb_beta: float
    output_dir: str
    top_k: int
    min_prob: float


assert set(_DEFAULTS) == {f.name for f in fields(Config)}, (
    "_DEFAULTS keys must match Config fields"
)


def load_config(path: Union[str, Path]) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    merged = {**_DEFAULTS, **raw}
    allowed = {f.name for f in fields(Config)}
    merged = {k: v for k, v in merged.items() if k in allowed}
    cfg = Config(**merged)
    if not 0.0 < cfg.val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {cfg.val_split}")
    if cfg.image_size <= 0:
        raise ValueError(f"image_size must be positive, got {cfg.image_size}")
    if not 0.0 <= cfg.cb_beta < 1.0:
        raise ValueError(f"cb_beta must be in [0, 1), got {cfg.cb_beta}")
    return cfg
