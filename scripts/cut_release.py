#!/usr/bin/env python3
"""One-command release script for synth-hillclimb.

Exports the Apache-2.0 hillclimb-cli/ package slice from a source repository,
bumps the package version, commits and tags the release on the target branch,
builds and bundles the native hillclimb-daemon binary via mini-build and
`uv build --wheel`, generates cohort.json and publish-manifest.json, verifies
all invariants with verify_publish.py, pushes to origin, creates the GitHub
Release, and dispatches the zero-click OIDC publish.yml workflow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from typing import Any
import zipfile

from verify_publish import COHORT_SCHEMA, MANIFEST_SCHEMA, PACKAGE_NAME, REPOSITORY, require

REPO_ROOT = Path(__file__).resolve().parents[1]
MINI_BUILD = Path.home() / ".synth-dev" / "bin" / "mini-build"
EXCLUDED_EXPORT_NAMES = frozenset(
    {
        ".build",
        ".DS_Store",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "dist",
    }
)
EXCLUDED_EXPORT_SUFFIXES = (".egg-info", ".pyc", ".pyo")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _run(
    cmd: list[str],
    *,
    cwd: Path = REPO_ROOT,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    print(f"+ ({cwd}) {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=capture,
        check=True,
    )


def _validate_version(version: str, channel: str) -> str:
    pattern = r"\d+\.\d+\.\d+" + (r"\.dev\d+" if channel == "nightly" else "")
    require(
        re.fullmatch(pattern, version) is not None,
        f"version {version!r} does not match channel {channel!r}",
    )
    return f"v{version}"


def _host_mini_build_platform() -> str:
    system = platform.system()
    if system == "Darwin":
        return "macos"
    if system == "Linux":
        return "linux"
    raise RuntimeError(f"Unsupported host OS for native wheel build: {system}")


def _export_cli_slice(source_cli: Path, dest_cli: Path) -> None:
    require(source_cli.is_dir(), f"source hillclimb-cli directory not found: {source_cli}")
    if dest_cli.exists():
        shutil.rmtree(dest_cli)

    def _ignore(_directory: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            if name in EXCLUDED_EXPORT_NAMES or name.endswith(EXCLUDED_EXPORT_SUFFIXES):
                ignored.add(name)
        return ignored

    shutil.copytree(source_cli, dest_cli, ignore=_ignore, symlinks=False)


def _bump_pyproject_version(pyproject_path: Path, version: str) -> None:
    require(pyproject_path.is_file(), f"missing pyproject.toml: {pyproject_path}")
    raw = pyproject_path.read_text(encoding="utf-8")
    parsed = tomllib.loads(raw)
    project = parsed.get("project", {})
    require(project.get("name") == PACKAGE_NAME, f"expected package name {PACKAGE_NAME!r}")
    require(project.get("license") == "Apache-2.0", "expected Apache-2.0 license in pyproject.toml")

    updated, count = re.subn(
        r'(?m)^(version\s*=\s*)"[^"]+"(\s*)$',
        rf'\g<1>"{version}"\g<2>',
        raw,
        count=1,
    )
    require(count == 1, f"failed to update version field in {pyproject_path}")
    pyproject_path.write_text(updated, encoding="utf-8")

    reparsed = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    require(reparsed["project"]["version"] == version, "pyproject.toml version bump mismatch")


def _build_daemon_binary(source_repo: Path) -> Path:
    require(MINI_BUILD.is_file(), f"mini-build launcher not found at {MINI_BUILD}")
    platform_arg = _host_mini_build_platform()
    _run(
        [
            str(MINI_BUILD),
            "build",
            "--platform",
            platform_arg,
            "--",
            "--release",
            "-p",
            "hillclimb-daemon",
        ],
        cwd=source_repo,
    )
    candidate = source_repo / "target" / "release" / "hillclimb-daemon"
    require(candidate.is_file(), f"built hillclimb-daemon binary not found at {candidate}")
    return candidate.resolve()


def _build_wheel(daemon_binary: Path, stage_dir: Path) -> Path:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["HILLCLIMB_DAEMON_BINARY"] = str(daemon_binary)
    _run(
        ["uv", "build", "--wheel", "--out-dir", str(stage_dir), str(REPO_ROOT / "hillclimb-cli")],
        cwd=REPO_ROOT,
        env=env,
    )
    wheels = sorted(stage_dir.glob("synth_hillclimb-*.whl"))
    require(len(wheels) == 1, f"expected exactly 1 wheel in {stage_dir}, found {len(wheels)}")
    return wheels[0]


def _write_manifests(wheel_path: Path, channel: str, version: str, stage_dir: Path) -> tuple[Path, Path, str]:
    with zipfile.ZipFile(wheel_path) as archive:
        native_manifest: dict[str, Any] = json.loads(archive.read("hillclimb/native/manifest.json"))

    require(native_manifest.get("sourceDirty") is False, "wheel was built from a dirty git tree")
    source_commit = native_manifest["sourceCommit"]
    source_files_sha256 = native_manifest["sourceFilesSha256"]
    native_sha256 = native_manifest["sha256"]
    native_member = native_manifest["nativeMember"]
    native_platform = native_manifest["platform"]
    wheel_sha256 = _sha256_file(wheel_path)

    cohort: dict[str, Any] = {
        "schema": COHORT_SCHEMA,
        "channel": channel,
        "version": version,
        "source": {
            "qualified_commit": source_commit,
            "source_files_sha256": source_files_sha256,
        },
        "artifacts": {
            "native": {
                "member": native_member,
                "platform": native_platform,
                "sha256": native_sha256,
            },
            "wheel": {
                "filename": wheel_path.name,
                "sha256": wheel_sha256,
            },
        },
    }
    cohort_canonical = json.dumps(cohort, sort_keys=True, separators=(",", ":")).encode("utf-8")
    cohort_sha256 = _sha256_bytes(cohort_canonical)
    cohort_path = stage_dir / "cohort.json"
    cohort_path.write_bytes(cohort_canonical + b"\n")

    publish_manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "channel": channel,
        "version": version,
        "cohort_sha256": cohort_sha256,
        "source_commit": source_commit,
        "source_files_sha256": source_files_sha256,
        "native_sha256": native_sha256,
        "wheel": {
            "filename": wheel_path.name,
            "sha256": wheel_sha256,
        },
    }
    manifest_bytes = (json.dumps(publish_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    manifest_path = stage_dir / "publish-manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    return cohort_path, manifest_path, manifest_sha256


def _ensure_clean_target_branch(branch: str) -> None:
    current = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture=True).stdout.strip()
    require(current == branch, f"expected current branch {branch!r}, got {current!r}")
    status = _run(["git", "status", "--porcelain"], capture=True).stdout.strip()
    require(status == "", f"working tree must be clean before cutting a release:\n{status}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cut and publish a synth-hillclimb release via zero-click PyPI OIDC.",
    )
    parser.add_argument(
        "--version",
        required=True,
        help="PEP 440 release version (e.g. 0.1.1 for stable, 0.1.1.dev1 for nightly).",
    )
    parser.add_argument(
        "--channel",
        choices=["stable", "nightly"],
        default="stable",
        help="Release channel (default: stable).",
    )
    parser.add_argument(
        "--source-repo",
        type=Path,
        default=Path("/Users/joshuapurtell/GitHub/hillclimb-beta"),
        help="Path to the hillclimb source repository containing hillclimb-cli/ and Rust crates.",
    )
    parser.add_argument(
        "--daemon-binary",
        type=Path,
        default=None,
        help="Optional pre-built hillclimb-daemon binary (skips mini-build if provided).",
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help="Directory for staged wheel and manifests (default: artifacts/release-v<version>).",
    )
    parser.add_argument(
        "--skip-dispatch",
        action="store_true",
        help="Create the GitHub Release without dispatching publish.yml.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned release steps without mutating git or GitHub.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tag = _validate_version(args.version, args.channel)
    branch = "nightly" if args.channel == "nightly" else "main"
    source_repo = args.source_repo.expanduser().resolve()
    source_cli = source_repo / "hillclimb-cli"
    stage_dir = (
        args.stage_dir.expanduser().resolve()
        if args.stage_dir is not None
        else REPO_ROOT / "artifacts" / f"release-{tag}"
    )

    require(source_repo.is_dir(), f"source repo does not exist: {source_repo}")
    require(source_cli.is_dir(), f"source hillclimb-cli/ does not exist: {source_cli}")
    if args.daemon_binary is not None:
        require(args.daemon_binary.expanduser().is_file(), f"daemon binary not found: {args.daemon_binary}")

    if args.dry_run:
        plan = {
            "action": "dry-run",
            "repository": REPOSITORY,
            "channel": args.channel,
            "branch": branch,
            "version": args.version,
            "tag": tag,
            "source_repo": str(source_repo),
            "stage_dir": str(stage_dir),
            "daemon_binary": str(args.daemon_binary) if args.daemon_binary else f"mini-build ({_host_mini_build_platform()})",
            "steps": [
                f"1. Verify clean {branch} branch and publisher environment",
                f"2. Export {source_cli} -> {REPO_ROOT / 'hillclimb-cli'} and bump version to {args.version}",
                f"3. Commit on {branch} ([skip ci]) and tag {tag}",
                "4. Build/locate hillclimb-daemon and run `uv build --wheel`",
                "5. Generate cohort.json + publish-manifest.json and verify assets",
                f"6. Push {branch} + {tag} to origin and run `verify_publish.py --check-git`",
                f"7. Create GitHub Release {tag}" + ("" if args.skip_dispatch else " and dispatch publish.yml"),
            ],
        }
        print(json.dumps(plan, indent=2))
        return

    _ensure_clean_target_branch(branch)
    existing_tag = _run(["git", "tag", "--list", tag], capture=True).stdout.strip()
    require(existing_tag == "", f"git tag {tag!r} already exists")
    _run([sys.executable, str(REPO_ROOT / "scripts" / "check_publisher_environment.py")])

    _export_cli_slice(source_cli, REPO_ROOT / "hillclimb-cli")
    _bump_pyproject_version(REPO_ROOT / "hillclimb-cli" / "pyproject.toml", args.version)

    _run(["git", "add", "hillclimb-cli"])
    _run(
        [
            "git",
            "commit",
            "-m",
            f"feat(release): export Apache-2.0 {PACKAGE_NAME} {tag} package [skip ci]",
        ]
    )
    _run(["git", "tag", tag])

    daemon_binary = (
        args.daemon_binary.expanduser().resolve()
        if args.daemon_binary is not None
        else _build_daemon_binary(source_repo)
    )
    wheel_path = _build_wheel(daemon_binary, stage_dir)
    cohort_path, manifest_path, manifest_sha256 = _write_manifests(
        wheel_path, args.channel, args.version, stage_dir
    )

    # Pre-push asset verification (manifest, cohort, wheel, native binary).
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "verify_publish.py"),
            "--assets",
            str(stage_dir),
            "--tag",
            tag,
            "--channel",
            args.channel,
            "--manifest-sha256",
            manifest_sha256,
        ]
    )

    _run(["git", "push", "origin", branch, tag])

    # Post-push full verification including origin/<branch> ancestry and tag tree hashes.
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "verify_publish.py"),
            "--assets",
            str(stage_dir),
            "--tag",
            tag,
            "--channel",
            args.channel,
            "--manifest-sha256",
            manifest_sha256,
            "--check-git",
        ]
    )

    _run(
        [
            "gh",
            "release",
            "create",
            tag,
            str(wheel_path),
            str(cohort_path),
            str(manifest_path),
            "--repo",
            REPOSITORY,
            "--title",
            f"{PACKAGE_NAME} {tag}",
            "--notes",
            f"Release {PACKAGE_NAME} {tag} ({args.channel}).\n\nManifest SHA-256: `{manifest_sha256}`",
        ]
    )

    if not args.skip_dispatch:
        _run(
            [
                "gh",
                "workflow",
                "run",
                "publish.yml",
                "--repo",
                REPOSITORY,
                "--ref",
                "main",
                "-f",
                f"channel={args.channel}",
                "-f",
                f"release_tag={tag}",
                "-f",
                f"manifest_sha256={manifest_sha256}",
            ]
        )

    print(
        json.dumps(
            {
                "status": "released",
                "tag": tag,
                "channel": args.channel,
                "wheel": wheel_path.name,
                "manifest_sha256": manifest_sha256,
                "dispatched": not args.skip_dispatch,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
