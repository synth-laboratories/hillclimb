#!/usr/bin/env python3
"""Deterministic identity of the files consumed by the hillclimb Rust workspace build."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = frozenset({"target", ".venv", "__pycache__", "node_modules", ".git"})


def _collect_tree(directory: Path, paths: set[Path]) -> None:
    if not directory.is_dir():
        return
    for candidate in directory.rglob("*"):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        if any(part in EXCLUDED_PARTS for part in candidate.parts):
            continue
        paths.add(candidate)


def snapshot(root: Path = ROOT) -> dict[str, Any]:
    """Return canonical file SHA-256 map and aggregate digest for hillclimb Rust inputs."""
    assert root.is_dir()
    paths: set[Path] = set()
    for fixed in ("Cargo.toml", "Cargo.lock", "rust-toolchain", "rust-toolchain.toml"):
        candidate = root / fixed
        if candidate.is_file() and not candidate.is_symlink():
            paths.add(candidate)
    for directory in (".cargo", "protocols", "optimizers"):
        _collect_tree(root / directory, paths)
    for crate_dir in sorted(root.glob("hillclimb-*")):
        _collect_tree(crate_dir, paths)
    crates_parent = root / "crates"
    if crates_parent.is_dir():
        for crate_dir in sorted(crates_parent.glob("hillclimb-*")):
            _collect_tree(crate_dir, paths)

    files = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert len(digest) == 64
    return {"sha256": digest, "files": files}


if __name__ == "__main__":
    print(json.dumps(snapshot(), sort_keys=True))
