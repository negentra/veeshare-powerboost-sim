"""Run manifest: provenance record for one full simulation run."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Optional


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def package_versions() -> Dict[str, str]:
    """Best-effort package version snapshot for the simulation stack."""
    names = [
        "numpy",
        "pandas",
        "scipy",
        "networkx",
        "simpy",
        "scikit-learn",
        "xgboost",
        "torch",
        "matplotlib",
        "pyyaml",
        "jinja2",
    ]
    out: Dict[str, str] = {}
    for n in names:
        try:
            mod_name = n.replace("-", "_")
            if n == "scikit-learn":
                mod_name = "sklearn"
            if n == "pyyaml":
                mod_name = "yaml"
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            out[n] = str(ver)
        except Exception:
            out[n] = "missing"
    return out


def build_manifest(
    *,
    run_mode: str,
    seed_root: int,
    config_paths: Dict[str, Path],
    output_paths: Dict[str, Path],
    wall_time_seconds: float,
    validation_passed: bool,
    git_commit: Optional[str] = None,
) -> dict:
    """Build a dict suitable for JSON serialisation as the run manifest."""
    return {
        "run_id": str(uuid.uuid4()),
        "run_mode": run_mode,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed_root": seed_root,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit or "untracked",
        "package_versions": package_versions(),
        "input_config_hashes": {
            name: _sha256_file(p) for name, p in config_paths.items() if p.exists()
        },
        "output_file_hashes": {
            name: _sha256_file(p) for name, p in output_paths.items() if p.exists()
        },
        "wall_time_seconds": float(wall_time_seconds),
        "validation_passed": bool(validation_passed),
        "notes": (
            "All outputs are simulation-derived; no empirical field data "
            "was used."
        ),
    }


def write_manifest(manifest: dict, out_path: Path) -> None:
    """Write the manifest as pretty JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit UTF-8: Windows default locale may not handle non-ASCII.
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False, ensure_ascii=False)
