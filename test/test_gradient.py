import torch
from ..src.model import MNISTNet


def test_gradient():
    x = torch.rand(28,28,requires_grad=True)
    model = MNISTNet()
    y = model(x)
    assert y.grad is not None