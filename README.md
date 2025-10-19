# mnisnet

mnisnet is a small PyTorch-based project that provides a MNIST example model and a simple training entry point. The code exposes a MNISTNet model (src.model.MNISTNet) and a training helper (src.train.train) intended for tests and quick experiments.

## Features
- MNISTNet model class (src.model.MNISTNet)
- Simple training entry point (src.train.train) that returns a trained model object
- Tests that validate shapes, gradients and a minimal training loop

## Requirements
- Python 3.8+
- PyTorch (version compatible with your system)
- pytest for running tests

Install typical dependencies (adjust versions as needed):
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# If you have a requirements.txt, use:
# pip install -r requirements.txt
pip install torch torchvision pytest
```

## Running tests
From the project root:
```bash
pytest -q
```

If tests use package imports (e.g. `from src.model import MNISTNet`) ensure you run pytest from the project root so the package layout is preserved.

## Quick usage
Start a Python REPL or a small script from the project root:

```python
from src.train import train
from src.model import MNISTNet

# run default training (returns a model instance)
model = train()
# use model.forward or model(...) for inference
print(hasattr(model, "forward"))  # should be True
```

## Project layout
- src/               — package containing model and training code
  - model.py         — contains MNISTNet
  - train.py         — contains train() helper
  - __init__.py      — exposes MNISTNet and train for convenience
- test/              — test suite (uses src.* imports)
- README.md          — this file

## Notes
- Tests expect the package layout to be importable from the project root. If you see import errors for `src`, ensure you are running commands from the repository root and that the virtual environment has needed packages installed.
- The README intentionally avoids implementation specifics; consult src/model.py and src/train.py for details.

## Contributing
Contributions are welcome. Keep changes small, include tests for new functionality, and follow the existing code style.

## License
Add a suitable LICENSE file or header to indicate the intended license for this project.
