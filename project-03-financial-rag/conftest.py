"""
Put the project root on sys.path so `from src... import ...` resolves.

Without this, `pytest tests/` works from inside the project directory (the
rootdir lands on sys.path) but `pytest project-0N-.../tests` from the repo
root fails collection with ModuleNotFoundError. Same command, different cwd,
different result -- which is exactly the kind of thing that only shows up in
someone else's CI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
