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
