from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from koi.config import Config
from koi.export_onnx import export
from koi.predict import _load_torch_model
from koi.train import train


def _cfg(data_dir: Path, out_dir: Path) -> Config:
    return Config(
        data_dir=str(data_dir), backbone="mobilenetv3_large_100", image_size=64,
        batch_size=4, epochs=1, lr=1e-3, weight_decay=0.0, label_smoothing=0.0,
        val_split=0.4, seed=0, num_workers=0, patience=5, output_dir=str(out_dir),
        top_k=0, min_prob=0.0,
    )


def test_export_matches_torch_outputs(synthetic_dataset: Path, tmp_path: Path):
    out = tmp_path / "runs"
    result = train(_cfg(synthetic_dataset, out), pretrained=False)
    onnx_path = tmp_path / "model.onnx"
    export(result["checkpoint"], onnx_path)
    assert onnx_path.exists()

    model, image_size, _ = _load_torch_model(result["checkpoint"])
    x = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        torch_out = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    assert np.allclose(torch_out, onnx_out, atol=1e-3)
