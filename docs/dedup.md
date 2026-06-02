# Duplicate-Removal Pipeline

Find and remove near-duplicate images from the dataset **before training**, so
the model doesn't over-count repeated photos. Lives in `koi/dedup.py`.

## How it works

For each breed folder, every image is reduced to a **perceptual hash**
(difference hash / dHash) — a 64-bit fingerprint computed from a downscaled
grayscale version. Two images are treated as "the same" when their hashes differ
by no more than `--threshold` bits (Hamming distance). Because the hash is based
on overall structure rather than exact bytes, it catches **resized, recompressed
and lightly edited copies**, not just identical files.

Images that match are grouped (a group can have 2+ members) and **copied** to a
review folder. Optionally, with `--apply`, all but one image per group is
**deleted** from the dataset.

- Scope: duplicates are detected **within each breed folder only**.
- The "diff function" is the `(hash_fn, threshold)` pair — both are configurable.

## Requirements

Already covered by `requirements.txt` (`pillow`, `numpy`). No extra install.

## Quick start

```bash
# 1. Dry run — copies duplicates out for review, deletes nothing
python -m koi.dedup --config config.yaml --threshold 3

# 2. Review what it found
ls duplicate/

# 3. Prune — keep one image per group, delete the rest (copies remain as backup)
python -m koi.dedup --config config.yaml --threshold 3 --apply
```

`--config` supplies the dataset location via its `data_dir`. To point at a
different folder directly, use `--data-dir /path/to/data` (overrides the config).

## Output layout

Matching groups are copied here (originals are left in place):

```
duplicate/
  Kohaku/
    00001_dup_01_<imagename>.jpg
    00001_dup_02_<imagename>.jpg
    00002_dup_01_<imagename>.jpg
    00002_dup_02_<imagename>.jpg
    00002_dup_03_<imagename>.jpg
  Bekko/
    00001_dup_01_<imagename>.jpg
    ...
```

- `NNNNN` — duplicate-group number, per breed (`00001`, `00002`, …).
- `dup_II` — index of the image within its group (`01`, `02`, …).
- `<imagename>.<ext>` — the original filename.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `config.yaml` | Config file; its `data_dir` is used unless `--data-dir` is given. |
| `--data-dir` | (from config) | Dataset root (one folder per breed). Overrides the config. |
| `--out` | `duplicate` | Where matching groups are copied. |
| `--threshold` | `3` | Max Hamming distance to call two images the same. `0` = pixel-identical only; higher = looser. |
| `--hash-size` | `8` | Hash grid size; `8` → a 64-bit hash. |
| `--apply` | off | Delete all but one image per group from the dataset. Without it the run is a copy-only dry run. |

## Choosing a threshold

Higher thresholds catch more variants but over-group **low-texture breeds**
(e.g. solid-colour Muji), where distinct fish look alike to a perceptual hash.
On this dataset:

| `--threshold` | Behaviour |
|---|---|
| `0` | Only pixel-identical re-encodes. Zero false positives, misses resized copies. |
| `2`–`3` | Catches resized/recompressed copies. **Recommended starting point.** |
| `5`+ | Over-groups solid-colour breeds (observed: 20 distinct Muji collapsed into one group). |

Start at `3`, review `duplicate/`, and lower to `2` or `0` if you see unrelated
fish grouped together.

## Recommended workflow

1. **Dry run** at `--threshold 3` and inspect `duplicate/` — especially
   low-texture breeds (Muji, Hikari-muji).
2. **Tune** the threshold down if unrelated images are grouped.
3. Work on a **copy** of the data (or back it up) before the next step.
4. **`--apply`** to prune. The copies under `duplicate/` are kept as a backup,
   so you can restore anything that was removed in error.
5. Re-run training; the dataset now has one image per duplicate group.

> ⚠️ `--apply` deletes files. It is off by default. Always review a dry run
> first and keep a backup — the `duplicate/` folder is your safety net.

## Programmatic use

```python
from koi.dedup import DuplicateFinder

finder = DuplicateFinder("data", output_dir="duplicate", threshold=3)

groups = finder.find_duplicates()        # {breed: [[Path, ...], ...]}
finder.copy_duplicates(groups)           # copy out for review
summary = finder.run(apply=True)         # copy, then keep 1 / delete the rest
print(summary)
# {'groups': N, 'duplicate_images': M, 'copied': M, 'removed': K, 'by_breed': {...}}
```

Swap the similarity definition by passing a different hash function:

```python
DuplicateFinder("data", hash_fn=my_hash_fn, threshold=4)
# my_hash_fn(image: PIL.Image, hash_size: int) -> np.ndarray  # 1-D bool array
```
