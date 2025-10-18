import torch
from ..src.model import L2Norm


def test_gradient():
    x = torch.rand(1,28*28,requires_grad=True)
    model = L2Norm().to('cpu')
    y = model(x).sum()
    y.backward()
    assert y.grad is  None