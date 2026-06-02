from pathlib import Path

from koi.config import Config
from koi.data import load_classes
from koi.train import train


def _cfg(data_dir: Path, out_dir: Path) -> Config:
    return Config(
        data_dir=str(data_dir), backbone="mobilenetv3_large_100", image_size=64,
        batch_size=4, epochs=1, lr=1e-3, weight_decay=0.0, label_smoothing=0.0,
        val_split=0.4, seed=0, num_workers=0, patience=5, cb_beta=0.999,
        output_dir=str(out_dir),
        top_k=0, min_prob=0.0,
    )


def test_train_writes_checkpoint_and_classes(synthetic_dataset: Path, tmp_path: Path):
    out = tmp_path / "runs"
    cfg = _cfg(synthetic_dataset, out)
    result = train(cfg, pretrained=False)
    assert Path(result["checkpoint"]).exists()
    assert Path(result["classes"]).exists()
    assert load_classes(result["classes"]) == ["ginrin-kohaku", "kohaku", "sanke"]
    assert 0.0 <= result["best_val_f1"] <= 1.0
    assert 0.0 <= result["best_val_acc"] <= 1.0
