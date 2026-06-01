import torch

from koi.model import build_model


def test_build_model_output_shape():
    model = build_model("mobilenetv3_large_100", num_classes=4, pretrained=False)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 4)
