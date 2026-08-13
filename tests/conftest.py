"""Make bare `pytest` work: import insect_gap9 from src/ and the paper scripts from
scripts/ -- the same layout every SLURM job exports via scripts/slurm/config.sh."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
