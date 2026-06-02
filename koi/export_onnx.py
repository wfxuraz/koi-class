import argparse
from pathlib import Path

import torch

from koi.config import load_config
from koi.predict import _load_torch_model


def export(checkpoint_path, onnx_path) -> str:
    model, image_size, _ = _load_torch_model(checkpoint_path)
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    _inline_external_data(onnx_path)
    return str(onnx_path)


def _inline_external_data(onnx_path) -> None:
    """Collapse the model into a single self-contained .onnx file.

    On torch >= 2.6 the dynamo exporter writes weights to a sidecar
    `<name>.onnx.data` by default. If only the .onnx is copied elsewhere the
    model is unusable, so we load the weights back in and re-save inline, then
    delete the sidecar. The model (~90 MB) is well under the 2 GB protobuf limit.
    """
    import onnx

    path = Path(onnx_path)
    model = onnx.load(str(path))  # pulls in any external data sidecar
    onnx.save_model(model, str(path), save_as_external_data=False)
    for sidecar in path.parent.glob(path.name + ".data"):
        sidecar.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    checkpoint = args.checkpoint or str(Path(cfg.output_dir) / "best_model.pt")
    out = args.out or str(Path(cfg.output_dir) / "model.onnx")
    export(checkpoint, out)
    print(f"exported {out}")


if __name__ == "__main__":
    main()
