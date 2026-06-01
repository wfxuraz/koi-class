import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from koi.config import load_config
from koi.data import (
    KoiDataset,
    build_transforms,
    discover_classes,
    save_classes,
    stratified_split,
)
from koi.model import build_model


def _evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return correct / total if total else 0.0


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
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    ckpt_path = out_dir / "best_model.pt"
    best_acc = -1.0
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
        acc = _evaluate(model, val_loader, device)
        print(f"epoch {epoch + 1}/{cfg.epochs}  val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
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
            "best_val_acc": max(best_acc, 0.0)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    result = train(cfg)
    print(f"best_val_acc={result['best_val_acc']:.4f}")
    print(f"checkpoint={result['checkpoint']}")


if __name__ == "__main__":
    main()
