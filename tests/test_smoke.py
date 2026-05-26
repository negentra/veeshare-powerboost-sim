"""End-to-end smoke test: run --mode smoke and verify outputs.

Marked as slow; skip with `pytest -m 'not slow'`.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.slow
def test_smoke_end_to_end():
    """Run python run.py --mode smoke and verify exit code + key files."""
    out_dir = ROOT / "outputs_test_smoke"
    shutil.rmtree(out_dir, ignore_errors=True)
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "run.py"),
                "--mode",
                "smoke",
                "--out-dir",
                str(out_dir),
            ],
            cwd=str(ROOT), capture_output=True, text=True, timeout=300,
        )
        assert result.returncode == 0, \
            f"smoke run failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        headline = out_dir / "tables" / "headline_numbers.csv"
        assert headline.exists() and headline.stat().st_size > 0
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
