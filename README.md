# mnisnet

mnisnet is a small PyTorch example project that provides:
- MNISTNet model (src.model.MNISTNet)
- A simple training entry point (src.train.train) used by tests and examples

Quick goals:
- Provide a minimal, testable MNIST model and training loop.
- Keep tests runnable from the project root using both package-relative and absolute imports.

Requirements
- Python 3.8+
- PyTorch (cpu or cuda build matching your system)
- pytest

Setup (recommended)
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision pytest
```

Run tests
From the project root run:
```bash
pytest -q
```

Usage example
```python
from src.train import train
from src.model import MNISTNet

model = train()        # runs the simple training helper and returns a model
print(hasattr(model, "forward"))
```

Project layout
- src/
  - model.py        — MNISTNet implementation
  - train.py        — train() helper returning a model instance
  - __init__.py     — package exports and test-friendly aliasing
- test/             — pytest test suite
- README.md         — this file
- .gitignore        — ignores data, model artifacts, mlruns, etc.

Notes & troubleshooting
- Tests and examples import the package as src (e.g. `from src.model import MNISTNet`). The package root is configured to expose the internal `mnisnet.src` as the top-level `src` module so imports work when running pytest from the repository root.
- If you see "ModuleNotFoundError: No module named 'src'", make sure:
  - You run pytest from the project root (/Users/faiazamin/Documents/Projects/AI/mnisnet).
  - Your virtualenv is active and dependencies are installed.
- If needed, run pytest with increased verbosity to inspect import failures:
```bash
pytest -q -k "" -s
```

Contributing
- Keep changes small and add tests for new behavior.
- Run pytest locally before pushing changes.

License
- Add a LICENSE file to indicate the intended license for the project.
