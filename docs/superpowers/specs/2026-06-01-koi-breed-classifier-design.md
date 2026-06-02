# Koi Breed Classifier — Design

**Date:** 2026-06-01
**Status:** Approved

## Goal

Classify an image of a single koi carp into its breed, returning per-breed
probabilities rendered top-to-bottom as percentages (e.g. `80% Kohaku,
20% Ginrin-Kohaku`). No bounding boxes — the input is a single koi that
roughly fills the frame, so this is pure image classification.

## Constraints

- **License:** Model and weights must be **Apache-2.0**. This rules out
  Ultralytics YOLO (YOLOv5/v8/v11), which is AGPL-3.0.
- **Inference target:** CPU / edge — speed matters. Favor a lightweight
  backbone and provide an ONNX export path for `onnxruntime` inference.
- **Dataset:** ~10–30 breeds, a few hundred to ~1k images per breed
  ("many classes, moderate data").
- **Input layout:** one directory per breed, images inside
  (e.g. `data/kohaku/*.jpg`, `data/ginrin-kohaku/*.jpg`).

## Model choice

Use **timm** (Apache-2.0) rather than a YOLO detection model. The task needs
no bounding boxes, so the YOLO architecture adds no value; timm gives
purpose-built classifiers with strong ImageNet-1k/21k pretrained weights,
which matter for fine-grained transfer learning on moderate data.

- **Default backbone:** `efficientnetv2_rw_s` (alt: `tf_efficientnetv2_s`) —
  strong fine-grained accuracy, reasonable CPU speed, Apache-2.0 pretrained
  weights.
- **Speed fallback:** `mobilenetv3_large_100`.
- Backbone is config-driven so backbones can be swapped and compared.

Koi breed classification is fine-grained (breeds differ by subtle scale/pattern
features, not gross shape), so transfer learning from strong pretrained weights
plus good augmentation is the primary driver of accuracy.

## Pipeline / data flow

```
data/<breed_name>/*.jpg
      │  (auto-detect classes, stratified train/val split)
      ▼
augmented batches ──► timm model (pretrained backbone + new head)
      │                      │ cross-entropy + label smoothing
      ▼                      ▼
   val metrics        best_model.pt  +  classes.json
                             │
            ┌────────────────┴───────────────┐
            ▼                                 ▼
      predict.py                       export_onnx.py ──► model.onnx
   (image → softmax →                  (CPU inference via onnxruntime)
    sorted percentages)
```

## Components (files)

- **`config.yaml`** — data dir, backbone name, image size (default 224),
  batch size, epochs, learning rate, train/val split ratio, output top-k /
  threshold.
- **`data.py`** — builds dataset from folder structure, auto-detects class
  names, performs stratified train/val split, applies augmentation (resize,
  horizontal flip, color jitter, random erasing — important for fine-grained
  robustness), and persists the class-index mapping to `classes.json`.
- **`model.py`** — `timm.create_model(backbone, pretrained=True,
  num_classes=N)`.
- **`train.py`** — transfer-learning loop, validation accuracy, label
  smoothing, early stopping, saves best checkpoint + `classes.json`.
  Auto-detects CPU/GPU.
- **`predict.py`** — accepts a single image or a folder; outputs breeds sorted
  high→low with percentages (top-k or all-above-threshold). Supports both the
  PyTorch checkpoint and the ONNX model.
- **`export_onnx.py`** — exports the trained model to `model.onnx` for
  CPU/edge inference via `onnxruntime`.
- **`requirements.txt`** — torch, timm, onnx, onnxruntime, pillow, pyyaml,
  scikit-learn.

## Output format

Softmax over classes, sorted descending, e.g.:

```
Kohaku           80.4%
Ginrin-Kohaku    14.2%
Sanke             5.4%
```

Configurable: show top-k or all classes above a probability threshold.

## Error handling

- Validate the data directory exists and is non-empty.
- Warn on classes with too few images (overfitting / split risk).
- Skip and log corrupt/unreadable images rather than crashing.
- `predict.py` must load the same `classes.json` the model was trained with;
  enforce that the model's output dimension matches the class count.
- Auto-detect device (CPU/GPU) for both training and inference.

## Testing

Smoke test on a tiny synthetic dataset (a few fake class folders with a couple
of images each):

- One training epoch runs end-to-end and saves a checkpoint + `classes.json`.
- `predict.py` returns probabilities that are sorted descending and sum to
  ~100%.
- ONNX export loads and its outputs match the PyTorch model within tolerance.

## Out of scope (YAGNI)

- Object detection / cropping (input is already a single framed koi).
- Multi-fish images / pond scenes.
- Web service / API (script-based for now).
