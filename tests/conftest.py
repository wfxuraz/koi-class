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
