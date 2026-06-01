from pathlib import Path

from koi.config import Config
from koi.export_onnx import export
from koi.predict import predict_image_onnx
from koi.train import train


def _cfg(data_dir: Path, out_dir: Path) -> Config:
    return Config(
        data_dir=str(data_dir), backbone="mobilenetv3_large_100", image_size=64,
        batch_size=4, epochs=1, lr=1e-3, weight_decay=0.0, label_smoothing=0.0,
        val_split=0.4, seed=0, num_workers=0, patience=5, output_dir=str(out_dir),
        top_k=0, min_prob=0.0,
    )


def test_predict_onnx_normalized_sorted(synthetic_dataset: Path, tmp_path: Path):
    out = tmp_path / "runs"
    result = train(_cfg(synthetic_dataset, out), pretrained=False)
    onnx_path = tmp_path / "model.onnx"
    export(result["checkpoint"], onnx_path)
    sample = next((synthetic_dataset / "kohaku").iterdir())
    ranked = predict_image_onnx(sample, onnx_path, result["classes"], image_size=64)
    total = sum(p for _, p in ranked)
    assert abs(total - 1.0) < 1e-4
    assert ranked == sorted(ranked, key=lambda x: x[1], reverse=True)
