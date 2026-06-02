# Koi Breed Classifier — Training Data Report

- **Date:** 2026-06-02
- **Data path:** `/mnt/c/Users/s_dhakal/Documents/koi-breed/data/`
- **Scanned by:** PIL `verify()` over all files (read from Windows mount via WSL)

## 1. Summary

| Metric | Value |
|---|---|
| Total files in tree | 31,404 |
| Valid training images | 31,380 (all `.jpg`) |
| Non-image files | 24 (`.xlsx`, all inside `_bk/`) |
| Image color modes | RGB only (31,380 / 31,380) |
| Corrupt / truncated images | 0 |
| Breed class directories | 22 (excluding `_bk`) |

**Verdict:** image data is **clean** — no corruption, no odd color modes.

## 2. Do I need to normalize the data?

**No.** Normalization and resizing are handled in code, not in the files.

`koi/data.py` → `build_transforms()`:

- `Resize((image_size, image_size))` — 224×224 from `config.yaml`
- `ToTensor()`
- `Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))` — ImageNet stats
- Train-only augmentation: `RandomHorizontalFlip`, `ColorJitter`, `RandomErasing`

> Do **not** pre-normalize or pre-resize pixels — it would be applied twice.
> Mixed source dimensions are fine; the transform standardizes them.

## 3. Per-class image counts (high → low)

| Breed | Count | Note |
|---|---:|---|
| Kohaku | 5220 | majority — will dominate |
| Showa-sanshoku | 3393 | |
| Taisho-sanshoku | 3383 | |
| A-ginrin | 2335 | |
| Kawarigoi | 1822 | |
| Goshiki | 1683 | |
| Tancho | 1618 | |
| Doitsugoi | 1491 | |
| Shiroutsuri | 1424 | |
| B-ginrin | 1284 | |
| Koromo | 1036 | |
| Kujaku | 1015 | |
| Hikari-utsuri | 908 | |
| Hikari-moyo | 834 | |
| Shusui | 790 | |
| Kumonryu | 663 | |
| Hikari-muji | 632 | |
| Asagi | 557 | |
| Hiutsuri | 469 | |
| Bekko | 412 | |
| Muji | 391 | |
| **Kiutsuri** | **20** | **too few** |
| `_bk` | 24 | **not a breed** — 24 `.xlsx` files (bookkeeping) |

- **Total breed images:** 31,380
- **Imbalance ratio (real classes):** 5220 / 20 = **261:1**

## 4. Action items before training

### 🔴 Required — move `_bk` out of the data directory

`discover_classes()` treats every subdir as a class. `_bk` contains only `.xlsx`, so it would become a 23rd class with 0 images: a dead output neuron and a polluted `classes.json`.

```bash
mv "/mnt/c/Users/s_dhakal/Documents/koi-breed/data/_bk" \
   "/mnt/c/Users/s_dhakal/Documents/koi-breed/_bk"
```

### 🟠 Important — decide what to do about Kiutsuri (20 images)

With `val_split=0.2` → ~16 train / 4 val. The model cannot learn it and its accuracy will be pure noise. Options:

1. Collect more Kiutsuri images
2. Drop the class
3. Merge it into a related class (Kiutsuri = yellow Utsuri)

### 🟠 Important — handle overall class imbalance (261:1)

`train.py` currently uses plain `CrossEntropyLoss` with shuffled sampling and **no class weighting** → majority breeds (Kohaku, Showa, Taisho) are favored. Recommended fix (code change, not data change):

- class-weighted `CrossEntropyLoss`, or
- `torch.utils.data.WeightedRandomSampler`

### ⚪ Optional — check for duplicate images

Scraped koi photos often contain exact duplicates. If the same image lands in both train and val it inflates reported accuracy (data leakage). A hash-based dedup check is recommended.

### ⚪ Optional (performance) — pre-resize for faster I/O

Reading from `/mnt/c` (Windows mount) in WSL is very slow. On Colab the data is copied into the VM / Drive anyway. Pre-resizing to ~256px shrinks the dataset and speeds up training I/O. Speed optimization only — not required for correctness, since transforms already resize.

## 5. Bottom line

Nothing to normalize — the code does it. Data is clean.
Before training: **(1) move `_bk`**, **(2) make a call on Kiutsuri / imbalance**.
