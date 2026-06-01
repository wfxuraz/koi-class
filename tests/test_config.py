from pathlib import Path

from koi.config import Config, load_config


def test_load_config_reads_yaml(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "data_dir: mydata\n"
        "backbone: mobilenetv3_large_100\n"
        "image_size: 160\n"
        "batch_size: 8\n"
        "epochs: 3\n"
        "lr: 0.001\n"
        "weight_decay: 0.0\n"
        "label_smoothing: 0.0\n"
        "val_split: 0.25\n"
        "seed: 1\n"
        "num_workers: 0\n"
        "patience: 2\n"
        "output_dir: out\n"
        "top_k: 3\n"
        "min_prob: 0.0\n"
    )
    cfg = load_config(cfg_file)
    assert isinstance(cfg, Config)
    assert cfg.data_dir == "mydata"
    assert cfg.image_size == 160
    assert cfg.top_k == 3


def test_load_config_rejects_bad_val_split(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("val_split: 1.5\n")
    try:
        load_config(cfg_file)
        assert False, "expected ValueError"
    except ValueError:
        pass
