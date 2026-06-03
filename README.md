# Koi Breed Classifier

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wfxuraz/koi-class/blob/main/colab.ipynb)

Apache-2.0 timm classifier for koi breeds. Input = one koi per image. Output =
breeds sorted high→low with percentages.

## Install

CPU/edge target — install CPU-only torch to avoid ~5GB of CUDA wheels:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Data layout

One directory per breed, images inside:

```
data/
  kohaku/         *.jpg
  ginrin-kohaku/  *.jpg
  sanke/          *.jpg
```

## Check for duplicates (optional, before training)

Find near-duplicate images within each breed (perceptual hash + Hamming
threshold) and copy them out for review — originals are left untouched:

```bash
python -m koi.dedup --config config.yaml --out duplicate --threshold 3
```

Matching groups are copied to `duplicate/<breed>/<NNNNN>_<L>_<name>.<ext>`, where
images in one set share the number `NNNNN` and differ by letter (`A`, `B`, `C`, …).
`--threshold 0` matches only pixel-identical re-encodes; higher values catch
resized/recompressed copies but over-group low-texture breeds (e.g. solid-colour
Muji), so review the output and tune per dataset.

After reviewing, prune the dataset (keeps one image per group, deletes the rest;
the copies in `duplicate/` remain as a backup):

```bash
python -m koi.dedup --config config.yaml --threshold 3 --apply
```

Without `--apply` the command is a dry run — it only copies, never deletes.
See [docs/dedup.md](docs/dedup.md) for the full pipeline, options, and workflow.

## Split into train / test (optional)

`koi.train` carves its own in-memory validation split for early stopping, so
that set isn't a clean hold-out. To get a test set the model never sees, split
the data on disk first — stratified per breed, deterministic, copy by default
(originals untouched):

```bash
python -m koi.split_dataset --config config.yaml --out data_split --test-split 0.2
```

Writes `data_split/train/<breed>/...`, `data_split/test/<breed>/...`, and a
`split_summary.json`. Then point `data_dir` in `config.yaml` at `data_split/train`
for training and evaluate on `data_split/test` (see below). Use `--move` instead
of copying, `--overwrite` to reuse a non-empty output, or `--help` for all flags.

## Train

```bash
python -m koi.train --config config.yaml
```

Writes `runs/best_model.pt` and `runs/classes.json`, plus `runs/metrics.csv`
(per-epoch train/val loss, macro-F1, accuracy). Settings (backbone, image size,
epochs, learning rate, augmentation behavior, output top-k/threshold) live in
`config.yaml`.

## Predict (PyTorch)

```bash
python -m koi.predict path/to/image.jpg
python -m koi.predict path/to/folder/
```

Example output:

```
kohaku           80.4%
ginrin-kohaku    14.2%
sanke             5.4%
```

## Evaluate (confusion matrix + per-class metrics)

Run the trained checkpoint over a labelled set and report a confusion matrix and
per-class precision / recall / F1:

```bash
# on the held-out test split from koi.split_dataset (the honest number):
python -m koi.evaluate --data-dir data_split/test --run-dir runs

# or on the same in-memory val split the trainer used:
python -m koi.evaluate --config config.yaml --run-dir runs
```

Writes `runs/analysis/confusion_matrix.png`, `eval_report.txt`, and
`eval_report.json`. Requires the dataset to be present.

## Inspect a training run (dashboard)

Turn the logs/metrics of a finished run into charts and a model sanity check:

```bash
python -m koi.analyze_run --run-dir runs
```

Writes `runs/analysis/dashboard.png` — one figure with loss, accuracy/macro-F1,
class distribution, and (if `koi.evaluate` has run) the confusion matrix — plus
standalone charts and `health.json` (checkpoint loads, output shape,
PyTorch↔ONNX parity). Works from `metrics.csv` when present, otherwise parses
`train.log`; panels note when loss or the dataset is unavailable.

## Export for CPU/edge deployment (ONNX)

```bash
python -m koi.export_onnx --config config.yaml
```

Writes `runs/model.onnx` for `onnxruntime` inference. Use
`koi.predict.predict_image_onnx(...)` to run the exported model on CPU.

## Tests

```bash
python -m pytest
```

## License

Uses timm (Apache-2.0) backbones. No AGPL components (no Ultralytics YOLO).
