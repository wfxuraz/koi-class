"""Analyze a finished training run: parse logs, plot curves, sanity-check the model.

Usage:
    python -m koi.analyze_run --run-dir koi-breed/runs

Outputs (written to <run-dir>/analysis/):
    training_curves.png   macro-F1 + accuracy vs epoch
    class_distribution.png  per-breed image counts (log scale)
    lowest_recall.png     how often each breed appeared in the weak-class list
    health.json           machine-readable sanity-check results

The script does NOT need the original dataset. It reads the artifacts left in
<run-dir> (train.log, report.md, best_model.pt, model.onnx, classes.json) and
runs a forward pass on a synthetic image to confirm the model loads and that the
PyTorch and ONNX graphs agree.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EPOCH_RE = re.compile(r"epoch (\d+)/(\d+)\s+macro_f1=([\d.]+)\s+acc=([\d.]+)")
RECALL_RE = re.compile(r"([A-Za-z-]+)=([\d.]+)")
BEST_RE = re.compile(r"best_val_f1=([\d.]+)\s+best_val_acc=([\d.]+)")
COUNT_RE = re.compile(r"^\|\s*([A-Za-z-]+)\s*\|\s*(\d+)\s*\|")


def parse_log(log_path: Path):
    """Return list of {epoch, total, f1, acc} and Counter of weak-class appearances."""
    text = log_path.read_text(errors="replace")
    epochs, weak = [], Counter()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = EPOCH_RE.search(line)
        if m:
            epochs.append(
                {
                    "epoch": int(m.group(1)),
                    "total": int(m.group(2)),
                    "f1": float(m.group(3)),
                    "acc": float(m.group(4)),
                }
            )
            # the following line usually lists "lowest-recall classes"
            if i + 1 < len(lines) and "recall" in lines[i + 1]:
                for name, _ in RECALL_RE.findall(lines[i + 1].split(":", 1)[-1]):
                    weak[name] += 1
    best = None
    bm = BEST_RE.search(text)
    if bm:
        best = {"f1": float(bm.group(1)), "acc": float(bm.group(2))}
    return epochs, weak, best


def parse_counts(report_path: Path):
    counts = {}
    if report_path.exists():
        for line in report_path.read_text(errors="replace").splitlines():
            m = COUNT_RE.match(line)
            if m and m.group(1) not in ("Breed",):
                counts[m.group(1)] = int(m.group(2))
    return counts


def plot_curves(epochs, best, out: Path):
    if not epochs:
        return
    xs = [e["epoch"] for e in epochs]
    f1 = [e["f1"] for e in epochs]
    acc = [e["acc"] for e in epochs]
    best_ep = xs[int(np.argmax(f1))]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, f1, "o-", label="macro-F1", color="#1f77b4")
    ax.plot(xs, acc, "s-", label="accuracy", color="#ff7f0e")
    ax.axvline(best_ep, ls="--", color="green", alpha=0.6,
               label=f"best (epoch {best_ep})")
    for x, y in zip(xs, f1):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_title("Validation metrics per epoch")
    ax.set_ylim(min(f1 + acc) - 0.02, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_distribution(counts, out: Path):
    if not counts:
        return
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(names, vals, color="#4c72b0")
    bars[-1].set_color("#c44e52")  # smallest class in red
    ax.set_yscale("log")
    ax.set_ylabel("images (log scale)")
    ax.set_title(f"Class distribution — {len(counts)} breeds, "
                 f"imbalance {max(vals) // max(min(vals), 1)}:1")
    ax.tick_params(axis="x", rotation=90)
    for b, v in zip(bars, vals):
        ax.annotate(str(v), (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_weak(weak, out: Path):
    if not weak:
        return
    items = weak.most_common()
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(names, vals, color="#c44e52")
    ax.set_ylabel("epochs in the weakest-5 list")
    ax.set_title("How often each breed showed up as a low-recall class")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def sanity_check(run_dir: Path):
    """Load the checkpoint + ONNX and run a forward pass on a synthetic image."""
    import torch
    from koi.data import build_transforms, load_classes
    from koi.model import build_model

    result = {}
    classes = load_classes(run_dir / "classes.json")
    result["num_classes"] = len(classes)

    ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu",
                      weights_only=True)
    image_size = ckpt["image_size"]
    result["backbone"] = ckpt["backbone"]
    result["image_size"] = image_size
    result["classes_match"] = (ckpt["classes"] == classes)

    model = build_model(ckpt["backbone"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # deterministic synthetic RGB image so the run is reproducible
    from PIL import Image
    arr = (np.indices((image_size, image_size)).sum(0) % 256).astype("uint8")
    rgb = np.stack([arr, arr[::-1], arr.T], axis=-1)
    img = Image.fromarray(rgb, "RGB")
    tf = build_transforms(image_size, train=False)
    tensor = tf(img).unsqueeze(0)

    with torch.no_grad():
        logits_t = model(tensor)
        probs_t = torch.softmax(logits_t, dim=1)[0].numpy()
    result["output_dim"] = int(logits_t.shape[1])
    result["torch_softmax_sums_to_one"] = bool(abs(probs_t.sum() - 1.0) < 1e-4)

    onnx_path = run_dir / "model.onnx"
    if onnx_path.exists():
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_path),
                                        providers=["CPUExecutionProvider"])
            logits_o = sess.run(
                None, {sess.get_inputs()[0].name: tensor.numpy()})[0][0]
            result["onnx_torch_max_abs_diff"] = float(
                np.abs(logits_o - logits_t[0].numpy()).max())
            result["onnx_parity_ok"] = result["onnx_torch_max_abs_diff"] < 1e-3
        except Exception as exc:  # broken/incomplete export shouldn't crash analysis
            result["onnx_parity_ok"] = False
            result["onnx_error"] = str(exc).splitlines()[-1][:200]
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="koi-breed/runs")
    args = ap.parse_args()
    run = Path(args.run_dir)
    out = run / "analysis"
    out.mkdir(exist_ok=True)

    epochs, weak, best = parse_log(run / "train.log")
    counts = parse_counts(run / "report.md")

    plot_curves(epochs, best, out / "training_curves.png")
    plot_distribution(counts, out / "class_distribution.png")
    plot_weak(weak, out / "lowest_recall.png")

    print(f"parsed {len(epochs)} epochs, {len(counts)} classes")
    health = sanity_check(run)
    health["best"] = best
    (out / "health.json").write_text(json.dumps(health, indent=2))
    print(json.dumps(health, indent=2))
    print(f"wrote graphs + health.json to {out}/")


if __name__ == "__main__":
    main()
