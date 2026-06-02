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

Matching groups are copied to `duplicate/<breed>/<NNNNN>_dup_<II>_<name>.<ext>`.
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

## Train

```bash
python -m koi.train --config config.yaml
```

Writes `runs/best_model.pt` and `runs/classes.json`. Settings (backbone, image
size, epochs, learning rate, augmentation behavior, output top-k/threshold) live
in `config.yaml`.

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
