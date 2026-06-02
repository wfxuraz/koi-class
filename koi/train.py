import argparse
from pathlib import Path

import torch
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader

from koi.config import load_config
from koi.data import (
    KoiDataset,
    build_transforms,
    class_balanced_weights,
    class_counts,
    discover_classes,
    save_classes,
    stratified_split,
)
from koi.model import build_model


def _evaluate(model, loader, device, num_classes: int) -> dict:
    """Return macro-F1, accuracy, and per-class recall on the loader."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
    if not all_labels:
        return {"macro_f1": 0.0, "accuracy": 0.0, "recall": [0.0] * num_classes}
    labels_range = list(range(num_classes))
    macro_f1 = f1_score(all_labels, all_preds, labels=labels_range,
                        average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, labels=labels_range,
                          average=None, zero_division=0)
    correct = sum(int(p == t) for p, t in zip(all_preds, all_labels))
    return {"macro_f1": float(macro_f1), "accuracy": correct / len(all_labels),
            "recall": [float(r) for r in recall]}


def train(cfg, pretrained: bool = True) -> dict:
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = discover_classes(cfg.data_dir)
    classes_path = out_dir / "classes.json"
    save_classes(classes, classes_path)

    train_s, val_s = stratified_split(cfg.data_dir, classes, cfg.val_split, cfg.seed)
    train_ds = KoiDataset(train_s, build_transforms(cfg.image_size, train=True))
    val_ds = KoiDataset(val_s, build_transforms(cfg.image_size, train=False))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers)

    model = build_model(cfg.backbone, len(classes), pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay)
    counts = class_counts(train_s, len(classes))
    weights = torch.tensor(class_balanced_weights(counts, cfg.cb_beta),
                           dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights,
                                          label_smoothing=cfg.label_smoothing)

    ckpt_path = out_dir / "best_model.pt"
    best_f1 = -1.0
    best_acc = 0.0
    epochs_no_improve = 0
    for epoch in range(cfg.epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        metrics = _evaluate(model, val_loader, device, len(classes))
        print(f"epoch {epoch + 1}/{cfg.epochs}  "
              f"macro_f1={metrics['macro_f1']:.4f}  acc={metrics['accuracy']:.4f}")
        worst = sorted(zip(classes, metrics["recall"]), key=lambda x: x[1])[:5]
        print("  lowest-recall classes: "
              + ", ".join(f"{name}={r:.2f}" for name, r in worst))
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_acc = metrics["accuracy"]
            epochs_no_improve = 0
            torch.save({"state_dict": model.state_dict(),
                        "backbone": cfg.backbone,
                        "image_size": cfg.image_size,
                        "classes": classes}, ckpt_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print(f"early stopping at epoch {epoch + 1}")
                break

    if not ckpt_path.exists():  # degenerate case: ensure something is saved
        torch.save({"state_dict": model.state_dict(),
                    "backbone": cfg.backbone,
                    "image_size": cfg.image_size,
                    "classes": classes}, ckpt_path)

    return {"checkpoint": str(ckpt_path), "classes": str(classes_path),
            "best_val_f1": max(best_f1, 0.0), "best_val_acc": best_acc}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = train(cfg)
    print(f"best_val_f1={result['best_val_f1']:.4f}  "
          f"best_val_acc={result['best_val_acc']:.4f}")
    print(f"checkpoint={result['checkpoint']}")


if __name__ == "__main__":
    main()
