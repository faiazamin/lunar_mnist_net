import torch
from ..src.model import MNISTNet


def test_shape():
    x = torch.rand(28,28,requires_grad=True)
    model = MNISTNet()
    y = model(x)
    assert y.shape == (28*28,10)
    