import json
from pathlib import Path

import torch

from koi.data import (
    KoiDataset,
    build_transforms,
    discover_classes,
    load_classes,
    save_classes,
    stratified_split,
)


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


def test_build_transforms_outputs_tensor_of_right_size(synthetic_dataset: Path):
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
