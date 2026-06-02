from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from koi.dedup import DuplicateFinder, dhash_bits


def _save(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr.astype(np.uint8)).convert("RGB").save(path, quality=95)


@pytest.fixture
def dup_dataset(tmp_path: Path) -> Path:
    """kohaku: two identical gradients + one distinct image. sanke: one image
    that matches a kohaku image (different breed -> must NOT be grouped)."""
    root = tmp_path / "data"
    horiz = np.tile(np.arange(64), (64, 1))          # strictly increasing left->right
    vert = horiz.T.copy()                            # increasing top->bottom (distinct dhash)

    k = root / "kohaku"
    k.mkdir(parents=True)
    _save(k / "a.jpg", horiz)
    _save(k / "a_copy.jpg", horiz)                   # duplicate of a.jpg
    _save(k / "b.jpg", vert)                         # distinct

    s = root / "sanke"
    s.mkdir()
    _save(s / "x.jpg", horiz)                        # same picture, other breed
    return root


def test_dhash_identical_images_match():
    arr = np.tile(np.arange(64), (64, 1))
    a = dhash_bits(Image.fromarray(arr.astype(np.uint8)).convert("L"))
    b = dhash_bits(Image.fromarray(arr.astype(np.uint8)).convert("L"))
    assert np.array_equal(a, b)
    assert a.shape == (64,)


def test_find_duplicates_groups_within_breed(dup_dataset: Path):
    dups = DuplicateFinder(dup_dataset).find_duplicates()
    assert set(dups) == {"kohaku"}                   # sanke single image -> skipped
    assert len(dups["kohaku"]) == 1                  # one duplicate group
    names = sorted(p.name for p in dups["kohaku"][0])
    assert names == ["a.jpg", "a_copy.jpg"]          # b.jpg excluded


def test_run_copies_with_expected_naming(dup_dataset: Path, tmp_path: Path):
    out = tmp_path / "duplicate"
    summary = DuplicateFinder(dup_dataset, output_dir=out).run()
    assert summary["groups"] == 1
    assert summary["duplicate_images"] == 2
    assert summary["copied"] == 2

    copied = sorted(p.name for p in (out / "kohaku").iterdir())
    assert copied == ["00001_dup_01_a.jpg", "00001_dup_02_a_copy.jpg"]
    # originals untouched
    assert (dup_dataset / "kohaku" / "a.jpg").exists()
    assert len(list((dup_dataset / "kohaku").iterdir())) == 3


def test_threshold_zero_only_exact_matches(dup_dataset: Path):
    dups = DuplicateFinder(dup_dataset, threshold=0).find_duplicates()
    # identical re-encodes still hash the same -> still grouped
    assert dups["kohaku"][0][0].name in {"a.jpg", "a_copy.jpg"}
