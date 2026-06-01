import argparse
from pathlib import Path

import torch

from koi.config import load_config
from koi.data import build_transforms, load_classes
from koi.model import build_model


def rank_predictions(probs, classes, top_k: int, min_prob: float):
    pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
    pairs = [(name, p) for name, p in pairs if p >= min_prob]
    if top_k and top_k > 0:
        pairs = pairs[:top_k]
    return pairs


def format_predictions(ranked) -> str:
    width = max((len(name) for name, _ in ranked), default=0)
    lines = [f"{name:<{width}}  {prob * 100:5.1f}%" for name, prob in ranked]
    return "\n".join(lines)


def _load_torch_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["backbone"], len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["image_size"], ckpt["classes"]


def predict_image(image_path, checkpoint_path, classes_path, top_k: int = 0,
                  min_prob: float = 0.0):
    model, image_size, ckpt_classes = _load_torch_model(checkpoint_path)
    classes = load_classes(classes_path)
    if classes != ckpt_classes:
        raise ValueError("classes.json does not match checkpoint classes")
    from PIL import Image
    tf = build_transforms(image_size, train=False)
    with Image.open(image_path) as img:
        tensor = tf(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].tolist()
    return rank_predictions(probs, classes, top_k, min_prob)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--classes", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    checkpoint = args.checkpoint or str(Path(cfg.output_dir) / "best_model.pt")
    classes = args.classes or str(Path(cfg.output_dir) / "classes.json")
    targets = [args.image]
    p = Path(args.image)
    if p.is_dir():
        targets = [str(x) for x in sorted(p.iterdir()) if x.is_file()]
    for target in targets:
        ranked = predict_image(target, checkpoint, classes, cfg.top_k, cfg.min_prob)
        print(f"\n{target}")
        print(format_predictions(ranked))


if __name__ == "__main__":
    main()
