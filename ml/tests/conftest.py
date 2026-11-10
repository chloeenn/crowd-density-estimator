from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "ShanghaiTech"

needs_data = pytest.mark.skipif(not RAW.exists(), reason="ShanghaiTech dataset not present")
