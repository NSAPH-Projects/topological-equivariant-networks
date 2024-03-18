# contents of tests/conftest.py
import sys
from pathlib import Path

# Assuming the structure is project_root/src and project_root/tests
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / "src"))
