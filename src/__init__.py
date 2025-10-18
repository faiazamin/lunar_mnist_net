"""src package initializer."""
from .model import MNISTNet  # noqa: F401
from .train import train    # noqa: F401

__all__ = ["MNISTNet", "train"]