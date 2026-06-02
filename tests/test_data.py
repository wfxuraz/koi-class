import json
from pathlib import Path

import torch

from koi.data import (
    KoiDataset,
    build_transforms,
    class_balanced_weights,
    class_counts,
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


def test_class_counts(synthetic_dataset: Path):
    classes = discover_classes(synthetic_dataset)
    train, val = stratified_split(synthetic_dataset, classes, 0.4, 0)
    counts = class_counts(train + val, len(classes))
    assert sum(counts) == 17  # 6 + 6 + 5
    assert len(counts) == 3


def test_class_balanced_weights_favor_rare_classes():
    # class 0 is rare, class 1 is common -> rare gets the larger weight
    weights = class_balanced_weights([10, 1000])
    assert weights[0] > weights[1]


def test_class_balanced_weights_equal_counts_are_uniform():
    weights = class_balanced_weights([100, 100, 100])
    assert all(abs(w - 1.0) < 1e-6 for w in weights)  # normalized to mean 1.0


def test_class_balanced_weights_empty_class_gets_zero():
    weights = class_balanced_weights([0, 50])
    assert weights[0] == 0.0
    assert weights[1] > 0.0


def test_save_and_load_classes_roundtrip(tmp_path: Path):
    classes = ["a", "b", "c"]
    path = tmp_path / "classes.json"
    save_classes(classes, path)
    assert json.loads(path.read_text()) == classes
    assert load_classes(path) == classes
