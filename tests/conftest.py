import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# The training/eval scripts live at the repo root, not in a package.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
