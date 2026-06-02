"""Evaluate a trained checkpoint on the held-out validation split.

This is the real "is the model working well?" check: it rebuilds the SAME
stratified val split used during training (same seed + val_split from
config.yaml), runs the checkpoint over it, and reports a confusion matrix plus
per-class precision / recall / F1.

Requires the dataset (config.data_dir) to be present. Run it where the data
lives (e.g. on Colab right after training, or locally if you have data/).

Usage:
    python -m koi.evaluate --config config.yaml --run-dir koi-breed/runs
    python -m koi.evaluate --data-dir /path/to/data   # evaluate every image in a folder
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from koi.config import load_config
from koi.data import (KoiDataset, build_transforms, discover_classes,
                      load_classes, stratified_split, _list_samples)
from koi.model import build_model


def _load_model(run_dir: Path):
    ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu",
                      weights_only=True)
    model = build_model(ckpt["backbone"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["image_size"], ckpt["classes"]


@torch.no_grad()
def _infer(model, loader, device):
    y_true, y_pred = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(labels.tolist())
    return np.array(y_true), np.array(y_pred)


def plot_confusion(cm, classes, out: Path):
    cmn = cm.astype(float) / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticklabels(classes, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Row-normalized confusion matrix (val split)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cmn[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5, color="white" if v > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-dir", default="koi-breed/runs")
    ap.add_argument("--data-dir", default=None,
                    help="evaluate EVERY image here instead of the val split")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config(args.config)
    run = Path(args.run_dir)
    out = run / "analysis"
    out.mkdir(exist_ok=True)

    model, image_size, classes = _load_model(run)
    if classes != load_classes(run / "classes.json"):
        raise ValueError("classes.json does not match checkpoint")

    data_dir = args.data_dir or cfg.data_dir
    tf = build_transforms(image_size, train=False)
    if args.data_dir:
        samples = _list_samples(data_dir, classes)  # whole folder
    else:
        _, samples = stratified_split(data_dir, classes, cfg.val_split, cfg.seed)
    print(f"evaluating {len(samples)} images over {len(classes)} classes")

    loader = DataLoader(KoiDataset(samples, tf), batch_size=args.batch_size,
                        num_workers=args.workers)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    y_true, y_pred = _infer(model, loader, device)

    report = classification_report(y_true, y_pred, labels=range(len(classes)),
                                   target_names=classes, digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, labels=range(len(classes)),
                                        target_names=classes, output_dict=True,
                                        zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    plot_confusion(cm, classes, out / "confusion_matrix.png")
    (out / "eval_report.txt").write_text(report)
    (out / "eval_report.json").write_text(json.dumps(report_dict, indent=2))

    acc = (y_true == y_pred).mean()
    macro_f1 = report_dict["macro avg"]["f1-score"]
    print(report)
    print(f"accuracy={acc:.4f}  macro_f1={macro_f1:.4f}")
    print(f"wrote confusion_matrix.png + eval_report.txt to {out}/")


if __name__ == "__main__":
    main()
