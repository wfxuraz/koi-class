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
