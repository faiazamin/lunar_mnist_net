"""Top-level mnisnet package.

Expose the internal subpackage `mnisnet.src` under the top-level name `src`
so tests that import `src.model` will find the package regardless of how
pytest adjusts sys.path when collecting package tests.
"""
from importlib import import_module
import sys

try:
    # import the local subpackage (mnisnet.src) and alias it as top-level 'src'
    _src_pkg = import_module(".src", __name__)
    # only set if not already present to avoid clobbering an existing top-level module
    sys.modules.setdefault("src", _src_pkg)
except Exception:
    # keep imports robust during collection; if subpackage doesn't exist, fail later with a clear error
    pass
