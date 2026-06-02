# Koi Breed Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a training + deployment pipeline that classifies a single-koi image into its breed, outputting per-breed percentages, using an Apache-2.0 timm backbone with an ONNX export for CPU/edge inference.

**Architecture:** Folder-per-breed dataset → stratified train/val split with augmentation → transfer-learned timm classifier (EfficientNetV2-S default) → best checkpoint + `classes.json` → ONNX export → `predict.py` runs either the PyTorch or ONNX model and prints breeds sorted high→low with percentages.

**Tech Stack:** Python, PyTorch, timm (Apache-2.0), onnx, onnxruntime, Pillow, PyYAML, scikit-learn, pytest.

---

## File Structure

- `requirements.txt` — pinned dependencies.
- `config.yaml` — all tunable settings (data dir, backbone, image size, training hyperparameters, output options).
- `koi/__init__.py` — package marker.
- `koi/config.py` — load + validate `config.yaml` into a dataclass.
- `koi/data.py` — class discovery, stratified split, datasets, transforms, `classes.json` persistence.
- `koi/model.py` — build a timm model from a backbone name + class count.
- `koi/train.py` — training/validation loop, checkpointing.
- `koi/predict.py` — load model (PyTorch or ONNX), classify an image/folder, format percentages.
- `koi/export_onnx.py` — export a checkpoint to ONNX.
- `tests/` — pytest suite + a tiny synthetic-dataset fixture.

Each module has one responsibility and is independently testable. CLI entry points (`train`, `predict`, `export_onnx`) are thin wrappers around library functions so the logic is unit-testable without spawning processes.

---

## Task 1: Project scaffold, git, dependencies, config

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `config.yaml`
- Create: `koi/__init__.py`
- Create: `koi/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Initialize git**

Run:
```bash
git init && git branch -M main
```
Expected: `Initialized empty Git repository`.

- [ ] **Step 2: Create `.gitignore`**

```gitignore
__pycache__/
*.pyc
.venv/
venv/
data/
runs/
*.onnx
*.pt
.pytest_cache/
```

- [ ] **Step 3: Create `requirements.txt`**

```text
# CPU/edge target: pull CPU-only torch to avoid ~5GB of CUDA wheels.
# Install with:  pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.2
torchvision>=0.17
timm>=1.0.7
onnx>=1.16
onnxruntime>=1.18
pillow>=10.0
pyyaml>=6.0
scikit-learn>=1.4
numpy>=1.26
pytest>=8.0
```

- [ ] **Step 4: Create `config.yaml`**

```yaml
data_dir: data
backbone: efficientnetv2_rw_s   # Apache-2.0 timm weights; fallback: mobilenetv3_large_100
image_size: 224
batch_size: 32
epochs: 20
lr: 0.0003
weight_decay: 0.0001
label_smoothing: 0.1
val_split: 0.2
seed: 42
num_workers: 4
patience: 5                     # early-stopping epochs without val-acc improvement
output_dir: runs
top_k: 0                        # 0 = show all classes; N>0 = show top N
min_prob: 0.0                   # only show classes with prob >= this (0.0 = no filter)
```

- [ ] **Step 5: Create `koi/__init__.py` and `tests/__init__.py`**

Both empty files:
```python
```

- [ ] **Step 6: Write the failing test for config loading**

`tests/test_config.py`:
```python
from pathlib import Path

from koi.config import Config, load_config


def test_load_config_reads_yaml(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "data_dir: mydata\n"
        "backbone: mobilenetv3_large_100\n"
        "image_size: 160\n"
        "batch_size: 8\n"
        "epochs: 3\n"
        "lr: 0.001\n"
        "weight_decay: 0.0\n"
        "label_smoothing: 0.0\n"
        "val_split: 0.25\n"
        "seed: 1\n"
        "num_workers: 0\n"
        "patience: 2\n"
        "output_dir: out\n"
        "top_k: 3\n"
        "min_prob: 0.0\n"
    )
    cfg = load_config(cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.data_dir == "mydata"
    assert cfg.image_size == 160
    assert cfg.top_k == 3


def test_load_config_rejects_bad_val_split(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("val_split: 1.5\n")
    try:
        load_config(cfg_file)
        assert False, "expected ValueError"
    except ValueError:
        pass
```

- [ ] **Step 7: Run the test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'koi.config'`.

- [ ] **Step 8: Implement `koi/config.py`**

```python
from dataclasses import dataclass, fields
from pathlib import Path

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
    output_dir: str
    top_k: int
    min_prob: float


def load_config(path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    merged = {**_DEFAULTS, **raw}
    allowed = {f.name for f in fields(Config)}
    merged = {k: v for k, v in merged.items() if k in allowed}
    cfg = Config(**merged)
    if not 0.0 < cfg.val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {cfg.val_split}")
    if cfg.image_size <= 0:
        raise ValueError(f"image_size must be positive, got {cfg.image_size}")
    return cfg
```

- [ ] **Step 9: Run the test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 10: Commit**

```bash
git add .gitignore requirements.txt config.yaml koi/__init__.py koi/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: project scaffold, config loader, dependencies"
```

---

## Task 2: Class discovery and stratified split

**Files:**
- Create: `koi/data.py`
- Create: `tests/conftest.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Create a synthetic-dataset fixture**

`tests/conftest.py`:
```python
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Three breeds, a handful of solid-color images each."""
    root = tmp_path / "data"
    specs = {
        "kohaku": ((220, 20, 20), 6),
        "ginrin-kohaku": ((20, 220, 20), 6),
        "sanke": ((20, 20, 220), 5),
    }
    for breed, (color, count) in specs.items():
        d = root / breed
        d.mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (64, 64), color).save(d / f"{breed}_{i}.jpg")
    return root
```

- [ ] **Step 2: Write the failing test for class discovery + split**

`tests/test_data.py`:
```python
from pathlib import Path

from koi.data import discover_classes, stratified_split


def test_discover_classes_sorted(synthetic_dataset: Path):
    classes = discover_classes(synthetic_dataset)
    assert classes == ["ginrin-kohaku", "kohaku", "sanke"]


def test_discover_classes_empty_dir_raises(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    try:
        discover_classes(tmp_path / "empty")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_stratified_split_covers_all_classes(synthetic_dataset: Path):
    classes = discover_classes(synthetic_dataset)
    train, val = stratified_split(synthetic_dataset, classes, val_split=0.4, seed=0)
    train_labels = {label for _, label in train}
    val_labels = {label for _, label in val}
    assert train_labels == set(range(len(classes)))
    assert val_labels == set(range(len(classes)))
    assert len(train) + len(val) == 17  # 6 + 6 + 5
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `python -m pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'koi.data'`.

- [ ] **Step 4: Implement discovery + split in `koi/data.py`**

```python
from pathlib import Path

from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_classes(data_dir) -> list[str]:
    root = Path(data_dir)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not classes:
        raise ValueError(f"No class subdirectories found in {root}")
    return classes


def _list_samples(data_dir, classes) -> list[tuple[str, int]]:
    root = Path(data_dir)
    samples: list[tuple[str, int]] = []
    for idx, cls in enumerate(classes):
        for p in (root / cls).iterdir():
            if p.suffix.lower() in IMG_EXTS:
                samples.append((str(p), idx))
    if not samples:
        raise ValueError(f"No images found under {root}")
    return samples


def stratified_split(data_dir, classes, val_split: float, seed: int):
    samples = _list_samples(data_dir, classes)
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    train_p, val_p, train_l, val_l = train_test_split(
        paths, labels, test_size=val_split, stratify=labels, random_state=seed
    )
    train = list(zip(train_p, train_l))
    val = list(zip(val_p, val_l))
    return train, val
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_data.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add koi/data.py tests/conftest.py tests/test_data.py
git commit -m "feat: class discovery and stratified train/val split"
```

---

## Task 3: Transforms, Dataset, and classes.json persistence

**Files:**
- Modify: `koi/data.py`
- Modify: `tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data.py`:
```python
import json

import torch

from koi.data import (
    KoiDataset,
    build_transforms,
    load_classes,
    save_classes,
)


def test_build_transforms_outputs_tensor_of_right_size(synthetic_dataset: Path):
    from koi.data import discover_classes, stratified_split

    classes = discover_classes(synthetic_dataset)
    train, _ = stratified_split(synthetic_dataset, classes, 0.4, 0)
    tf = build_transforms(image_size=96, train=False)
    ds = KoiDataset(train, tf)
    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 96, 96)
    assert isinstance(label, int)


def test_save_and_load_classes_roundtrip(tmp_path: Path):
    classes = ["a", "b", "c"]
    path = tmp_path / "classes.json"
    save_classes(classes, path)
    assert json.loads(path.read_text()) == classes
    assert load_classes(path) == classes
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_data.py -v`
Expected: FAIL with `ImportError: cannot import name 'KoiDataset'`.

- [ ] **Step 3: Implement transforms, dataset, persistence**

Append to `koi/data.py`:
```python
import json

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
            transforms.RandomErasing(p=0.25),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


class KoiDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            image = self.transform(img.convert("RGB"))
        return image, label


def save_classes(classes, path) -> None:
    from pathlib import Path
    Path(path).write_text(json.dumps(classes))


def load_classes(path) -> list[str]:
    from pathlib import Path
    return json.loads(Path(path).read_text())
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_data.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add koi/data.py tests/test_data.py
git commit -m "feat: transforms, dataset, classes.json persistence"
```

---

## Task 4: Build the timm model

**Files:**
- Create: `koi/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write the failing test**

`tests/test_model.py`:
```python
import torch

from koi.model import build_model


def test_build_model_output_shape():
    model = build_model("mobilenetv3_large_100", num_classes=4, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 4)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'koi.model'`.

- [ ] **Step 3: Implement `koi/model.py`**

```python
import timm
import torch.nn as nn


def build_model(backbone: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_model.py -v`
Expected: PASS. (Uses `pretrained=False` so no network download in tests.)

- [ ] **Step 5: Commit**

```bash
git add koi/model.py tests/test_model.py
git commit -m "feat: timm model builder"
```

---

## Task 5: Probability formatting (output rendering)

This is the user-facing "80% Kohaku / 20% Ginrin-Kohaku" logic. Isolated and pure so it is easy to test.

**Files:**
- Create: `koi/predict.py`
- Create: `tests/test_predict_format.py`

- [ ] **Step 1: Write the failing test**

`tests/test_predict_format.py`:
```python
from koi.predict import format_predictions, rank_predictions


def test_rank_predictions_sorts_descending():
    classes = ["a", "b", "c"]
    probs = [0.2, 0.7, 0.1]
    ranked = rank_predictions(probs, classes, top_k=0, min_prob=0.0)
    assert ranked[0] == ("b", 0.7)
    assert [name for name, _ in ranked] == ["b", "a", "c"]


def test_rank_predictions_top_k_and_threshold():
    classes = ["a", "b", "c"]
    probs = [0.2, 0.7, 0.1]
    assert rank_predictions(probs, classes, top_k=2, min_prob=0.0) == [
        ("b", 0.7), ("a", 0.2)
    ]
    assert rank_predictions(probs, classes, top_k=0, min_prob=0.15) == [
        ("b", 0.7), ("a", 0.2)
    ]


def test_format_predictions_renders_percentages():
    ranked = [("kohaku", 0.804), ("ginrin-kohaku", 0.142)]
    text = format_predictions(ranked)
    assert "kohaku" in text
    assert "80.4%" in text
    assert "14.2%" in text
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_predict_format.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'koi.predict'`.

- [ ] **Step 3: Implement formatting in `koi/predict.py`**

```python
def rank_predictions(probs, classes, top_k: int, min_prob: float):
    pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    pairs = [(name, p) for name, p in pairs if p >= min_prob]
    if top_k and top_k > 0:
        pairs = pairs[:top_k]
    return pairs


def format_predictions(ranked) -> str:
    width = max((len(name) for name, _ in ranked), default=0)
    lines = [f"{name:<{width}}  {prob * 100:5.1f}%" for name, prob in ranked]
    return "\n".join(lines)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_predict_format.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add koi/predict.py tests/test_predict_format.py
git commit -m "feat: prediction ranking and percentage formatting"
```

---

## Task 6: Training loop and checkpointing

**Files:**
- Create: `koi/train.py`
- Create: `tests/test_train.py`

- [ ] **Step 1: Write the failing end-to-end-ish training test**

`tests/test_train.py`:
```python
from pathlib import Path

from koi.config import Config
from koi.data import load_classes
from koi.train import train


def _cfg(data_dir: Path, out_dir: Path) -> Config:
    return Config(
        data_dir=str(data_dir), backbone="mobilenetv3_large_100", image_size=64,
        batch_size=4, epochs=1, lr=1e-3, weight_decay=0.0, label_smoothing=0.0,
        val_split=0.4, seed=0, num_workers=0, patience=5, output_dir=str(out_dir),
        top_k=0, min_prob=0.0,
    )


def test_train_writes_checkpoint_and_classes(synthetic_dataset: Path, tmp_path: Path):
    out = tmp_path / "runs"
    cfg = _cfg(synthetic_dataset, out)
    result = train(cfg, pretrained=False)
    assert Path(result["checkpoint"]).exists()
    assert Path(result["classes"]).exists()
    assert load_classes(result["classes"]) == ["ginrin-kohaku", "kohaku", "sanke"]
    assert 0.0 <= result["best_val_acc"] <= 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'koi.train'`.

- [ ] **Step 3: Implement `koi/train.py`**

```python
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from koi.config import load_config
from koi.data import (
    KoiDataset,
    build_transforms,
    discover_classes,
    save_classes,
    stratified_split,
)
from koi.model import build_model


def _evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total if total else 0.0


def train(cfg, pretrained: bool = True) -> dict:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = discover_classes(cfg.data_dir)
    classes_path = out_dir / "classes.json"
    save_classes(classes, classes_path)

    train_s, val_s = stratified_split(cfg.data_dir, classes, cfg.val_split, cfg.seed)
    train_ds = KoiDataset(train_s, build_transforms(cfg.image_size, train=True))
    val_ds = KoiDataset(val_s, build_transforms(cfg.image_size, train=False))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)

    model = build_model(cfg.backbone, len(classes), pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    ckpt_path = out_dir / "best_model.pt"
    best_acc = -1.0
    epochs_no_improve = 0
    for epoch in range(cfg.epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        acc = _evaluate(model, val_loader, device)
        print(f"epoch {epoch + 1}/{cfg.epochs}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
            torch.save({"state_dict": model.state_dict(),
                        "backbone": cfg.backbone,
                        "image_size": cfg.image_size,
                        "classes": classes}, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"early stopping at epoch {epoch + 1}")
                break

    if not ckpt_path.exists():  # degenerate case: ensure something is saved
        torch.save({"state_dict": model.state_dict(),
                    "backbone": cfg.backbone,
                    "image_size": cfg.image_size,
                    "classes": classes}, ckpt_path)

    return {"checkpoint": str(ckpt_path), "classes": str(classes_path),
            "best_val_acc": max(best_acc, 0.0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = train(cfg)
    print(f"best_val_acc={result['best_val_acc']:.4f}")
    print(f"checkpoint={result['checkpoint']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_train.py -v`
Expected: PASS (1 passed). May take a minute on CPU.

- [ ] **Step 5: Commit**

```bash
git add koi/train.py tests/test_train.py
git commit -m "feat: training loop with checkpointing and early stopping"
```

---

## Task 7: PyTorch inference end-to-end

**Files:**
- Modify: `koi/predict.py`
- Create: `tests/test_predict_torch.py`

- [ ] **Step 1: Write the failing test (depends on Task 6 checkpoint)**

`tests/test_predict_torch.py`:
```python
from pathlib import Path

from koi.config import Config
from koi.predict import predict_image
from koi.train import train


def _cfg(data_dir: Path, out_dir: Path) -> Config:
    return Config(
        data_dir=str(data_dir), backbone="mobilenetv3_large_100", image_size=64,
        batch_size=4, epochs=1, lr=1e-3, weight_decay=0.0, label_smoothing=0.0,
        val_split=0.4, seed=0, num_workers=0, patience=5, output_dir=str(out_dir),
        top_k=0, min_prob=0.0,
    )


def test_predict_image_returns_normalized_sorted_probs(synthetic_dataset: Path, tmp_path: Path):
    out = tmp_path / "runs"
    result = train(_cfg(synthetic_dataset, out), pretrained=False)
    sample = next((synthetic_dataset / "kohaku").iterdir())
    ranked = predict_image(sample, result["checkpoint"], result["classes"])
    total = sum(p for _, p in ranked)
    assert abs(total - 1.0) < 1e-4
    assert ranked == sorted(ranked, key=lambda x: x[1], reverse=True)
    assert {name for name, _ in ranked} == {"ginrin-kohaku", "kohaku", "sanke"}
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_predict_torch.py -v`
Expected: FAIL with `ImportError: cannot import name 'predict_image'`.

- [ ] **Step 3: Add inference to `koi/predict.py`**

Append to `koi/predict.py`:
```python
import argparse
from pathlib import Path

import torch

from koi.config import load_config
from koi.data import build_transforms, load_classes
from koi.model import build_model


def _load_torch_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["backbone"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["image_size"], ckpt["classes"]


def predict_image(image_path, checkpoint_path, classes_path, top_k: int = 0,
                  min_prob: float = 0.0):
    model, image_size, ckpt_classes = _load_torch_model(checkpoint_path)
    classes = load_classes(classes_path)
    if classes != ckpt_classes:
        raise ValueError("classes.json does not match checkpoint classes")
    from PIL import Image
    tf = build_transforms(image_size, train=False)
    with Image.open(image_path) as img:
        tensor = tf(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].tolist()
    return rank_predictions(probs, classes, top_k, min_prob)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--classes", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    checkpoint = args.checkpoint or str(Path(cfg.output_dir) / "best_model.pt")
    classes = args.classes or str(Path(cfg.output_dir) / "classes.json")
    targets = [args.image]
    p = Path(args.image)
    if p.is_dir():
        targets = [str(x) for x in sorted(p.iterdir()) if x.is_file()]
    for target in targets:
        ranked = predict_image(target, checkpoint, classes, cfg.top_k, cfg.min_prob)
        print(f"\n{target}")
        print(format_predictions(ranked))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_predict_torch.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add koi/predict.py tests/test_predict_torch.py
git commit -m "feat: PyTorch single-image and folder inference"
```

---

## Task 8: ONNX export (deployment artifact)

**Files:**
- Create: `koi/export_onnx.py`
- Create: `tests/test_export_onnx.py`

- [ ] **Step 1: Write the failing test**

`tests/test_export_onnx.py`:
```python
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from koi.config import Config
from koi.export_onnx import export
from koi.predict import _load_torch_model
from koi.train import train


def _cfg(data_dir: Path, out_dir: Path) -> Config:
    return Config(
        data_dir=str(data_dir), backbone="mobilenetv3_large_100", image_size=64,
        batch_size=4, epochs=1, lr=1e-3, weight_decay=0.0, label_smoothing=0.0,
        val_split=0.4, seed=0, num_workers=0, patience=5, output_dir=str(out_dir),
        top_k=0, min_prob=0.0,
    )


def test_export_matches_torch_outputs(synthetic_dataset: Path, tmp_path: Path):
    out = tmp_path / "runs"
    result = train(_cfg(synthetic_dataset, out), pretrained=False)
    onnx_path = tmp_path / "model.onnx"
    export(result["checkpoint"], onnx_path)
    assert onnx_path.exists()

    model, image_size, _ = _load_torch_model(result["checkpoint"])
    x = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        torch_out = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    assert np.allclose(torch_out, onnx_out, atol=1e-3)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_export_onnx.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'koi.export_onnx'`.

- [ ] **Step 3: Implement `koi/export_onnx.py`**

```python
import argparse
from pathlib import Path

import torch

from koi.config import load_config
from koi.predict import _load_torch_model


def export(checkpoint_path, onnx_path) -> str:
    model, image_size, _ = _load_torch_model(checkpoint_path)
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    return str(onnx_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    checkpoint = args.checkpoint or str(Path(cfg.output_dir) / "best_model.pt")
    out = args.out or str(Path(cfg.output_dir) / "model.onnx")
    export(checkpoint, out)
    print(f"exported {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_export_onnx.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add koi/export_onnx.py tests/test_export_onnx.py
git commit -m "feat: ONNX export with torch-parity test"
```

---

## Task 9: ONNX inference path in predict.py

**Files:**
- Modify: `koi/predict.py`
- Create: `tests/test_predict_onnx.py`

- [ ] **Step 1: Write the failing test**

`tests/test_predict_onnx.py`:
```python
from pathlib import Path

from koi.config import Config
from koi.export_onnx import export
from koi.predict import predict_image_onnx
from koi.train import train


def _cfg(data_dir: Path, out_dir: Path) -> Config:
    return Config(
        data_dir=str(data_dir), backbone="mobilenetv3_large_100", image_size=64,
        batch_size=4, epochs=1, lr=1e-3, weight_decay=0.0, label_smoothing=0.0,
        val_split=0.4, seed=0, num_workers=0, patience=5, output_dir=str(out_dir),
        top_k=0, min_prob=0.0,
    )


def test_predict_onnx_normalized_sorted(synthetic_dataset: Path, tmp_path: Path):
    out = tmp_path / "runs"
    result = train(_cfg(synthetic_dataset, out), pretrained=False)
    onnx_path = tmp_path / "model.onnx"
    export(result["checkpoint"], onnx_path)
    sample = next((synthetic_dataset / "kohaku").iterdir())
    ranked = predict_image_onnx(sample, onnx_path, result["classes"], image_size=64)
    total = sum(p for _, p in ranked)
    assert abs(total - 1.0) < 1e-4
    assert ranked == sorted(ranked, key=lambda x: x[1], reverse=True)
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/test_predict_onnx.py -v`
Expected: FAIL with `ImportError: cannot import name 'predict_image_onnx'`.

- [ ] **Step 3: Add ONNX inference to `koi/predict.py`**

Append to `koi/predict.py`:
```python
def predict_image_onnx(image_path, onnx_path, classes_path, image_size: int,
                       top_k: int = 0, min_prob: float = 0.0):
    import numpy as np
    import onnxruntime as ort
    from PIL import Image

    from koi.data import build_transforms, load_classes

    classes = load_classes(classes_path)
    tf = build_transforms(image_size, train=False)
    with Image.open(image_path) as img:
        tensor = tf(img.convert("RGB")).unsqueeze(0).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    logits = sess.run(None, {sess.get_inputs()[0].name: tensor})[0][0]
    exp = np.exp(logits - logits.max())
    probs = (exp / exp.sum()).tolist()
    return rank_predictions(probs, classes, top_k, min_prob)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/test_predict_onnx.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add koi/predict.py tests/test_predict_onnx.py
git commit -m "feat: ONNX inference path in predict"
```

---

## Task 10: README and full suite run

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

````markdown
# Koi Breed Classifier

Apache-2.0 timm classifier for koi breeds. Input = one koi per image. Output =
breeds sorted high→low with percentages.

## Install
```bash
pip install -r requirements.txt
```

## Data layout
```
data/
  kohaku/         *.jpg
  ginrin-kohaku/  *.jpg
  sanke/          *.jpg
```

## Train
```bash
python -m koi.train --config config.yaml
```
Writes `runs/best_model.pt` and `runs/classes.json`.

## Predict (PyTorch)
```bash
python -m koi.predict path/to/image.jpg
python -m koi.predict path/to/folder/
```

## Export for CPU/edge deployment (ONNX)
```bash
python -m koi.export_onnx --config config.yaml
```
Writes `runs/model.onnx` for `onnxruntime` inference.

## License
Uses timm (Apache-2.0) backbones. No AGPL components.
````

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest -v`
Expected: PASS (all tests).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: usage README"
```

---

## Self-Review Notes

- **Spec coverage:** data layout/discovery (Task 2), augmentation (Task 3), timm Apache-2.0 model (Task 4), output percentages incl. top-k/threshold (Task 5), training + checkpoint + classes.json + early stopping + device auto-detect (Task 6), PyTorch inference incl. folder + classes-match guard (Task 7), ONNX export (Task 8), ONNX/onnxruntime CPU inference (Task 9), README (Task 10). Corrupt-image skipping is the one spec item intentionally deferred — note below.
- **Deferred / follow-up:** Spec mentions "skip and log corrupt images" and "warn on classes with too few images." These are small robustness additions; add a Task 3.5 wrapper around `Image.open` and a count check in `discover_classes` if you want them in v1. Flagged rather than silently dropped.
- **Type consistency:** `build_model(backbone, num_classes, pretrained)`, checkpoint dict keys (`state_dict`, `backbone`, `image_size`, `classes`), `rank_predictions(probs, classes, top_k, min_prob)`, and `build_transforms(image_size, train)` are used identically across tasks.
- **No network in tests:** every test uses `pretrained=False`, so CI/offline runs don't download weights.
