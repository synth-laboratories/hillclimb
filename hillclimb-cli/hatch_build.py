"""Package already-qualified native hillclimb-daemon bytes; never silently invokes cargo."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PLATFORM_TAGS: dict[tuple[str, str], str] = {
    ("Darwin", "arm64"): "macosx_11_0_arm64",
    ("Linux", "x86_64"): "manylinux_2_28_x86_64",
    ("Linux", "aarch64"): "manylinux_2_28_aarch64",
}

DAEMON_BINARY_ENV = "HILLCLIMB_DAEMON_BINARY"
SOURCE_COMMIT_ENV = "HILLCLIMB_SOURCE_COMMIT"
NATIVE_MEMBER = "hillclimb/native/hillclimb-daemon"
NOTICE_FILES = ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.txt", "PROVENANCE.md")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _verify_optional_receipt(
    binary: Path,
    binary_sha256: str,
    expected_version: str,
    expected_platform: str,
) -> dict[str, Any]:
    receipt_path = Path(f"{binary}.build.json")
    if not receipt_path.exists():
        return {}
    if not receipt_path.is_file():
        raise RuntimeError(f"Native build receipt is not a regular file: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Failed to parse native build receipt {receipt_path}: {error}") from error
    if not isinstance(receipt, dict):
        raise RuntimeError(f"Native build receipt must be a JSON object: {receipt_path}")

    receipt_sha256 = receipt.get("binary_sha256")
    if not isinstance(receipt_sha256, str) or receipt_sha256 != binary_sha256:
        raise RuntimeError(
            f"Native binary SHA-256 ({binary_sha256}) does not match receipt ({receipt_sha256!r})"
        )

    receipt_version = receipt.get("version")
    if receipt_version is not None:
        accepted_versions = {expected_version, expected_version.replace(".dev", "-dev.")}
        if receipt_version not in accepted_versions:
            raise RuntimeError(
                f"Native receipt version {receipt_version!r} disagrees with Python version {expected_version!r}"
            )

    receipt_platform = receipt.get("platform")
    if receipt_platform is not None and receipt_platform != expected_platform:
        raise RuntimeError(
            f"Native receipt platform {receipt_platform!r} disagrees with wheel platform {expected_platform!r}"
        )

    return receipt


def _git_provenance(repo: Path, receipt: dict[str, Any]) -> tuple[str, bool, dict[str, str], str]:
    probe = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0 and probe.stdout.strip() == "true":
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip())
        raw_names = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", "-z", commit],
            cwd=repo,
        ).decode("utf-8").rstrip("\0")
        names = raw_names.split("\0") if raw_names else []
        source_files: dict[str, str] = {}
        for name in sorted(names):
            blob = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=repo)
            source_files[name] = _sha256_bytes(blob)
        source_files_sha256 = _sha256_bytes(
            json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        return commit, dirty, source_files, source_files_sha256

    fallback_commit = os.environ.get(SOURCE_COMMIT_ENV) or receipt.get("sourceCommit") or "unavailable"
    return str(fallback_commit), False, {}, _sha256_bytes(b"{}")


class NativeHook(BuildHookInterface):
    """Hatchling wheel hook bundling pre-qualified hillclimb-daemon native bytes."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if self.target_name != "wheel" or version == "editable":
            return

        configured = os.environ.get(DAEMON_BINARY_ENV, "").strip()
        if not configured:
            raise RuntimeError(
                f"{DAEMON_BINARY_ENV} must identify the pre-qualified hillclimb-daemon binary"
            )
        binary = Path(configured).expanduser()
        if not binary.is_file():
            raise RuntimeError(
                f"{DAEMON_BINARY_ENV} must point to an existing file: {binary}"
            )

        host = (platform.system(), platform.machine())
        if host not in PLATFORM_TAGS:
            raise RuntimeError(f"Unqualified wheel platform: {host}")
        target_tag = PLATFORM_TAGS[host]

        binary_sha256 = _sha256_bytes(binary.read_bytes())
        receipt = _verify_optional_receipt(
            binary,
            binary_sha256,
            self.metadata.version,
            target_tag,
        )

        root = Path(self.root).resolve()
        repo = root.parent if (root.parent / ".git").exists() else root
        source_commit, source_dirty, source_files, source_files_sha256 = _git_provenance(repo, receipt)

        scratch = root / ".build"
        scratch.mkdir(parents=True, exist_ok=True)
        manifest_path = scratch / "manifest.json"
        manifest_payload: dict[str, Any] = {
            "version": self.metadata.version,
            "sha256": binary_sha256,
            "nativeMember": NATIVE_MEMBER,
            "sourceCommit": source_commit,
            "sourceDirty": source_dirty,
            "sourceFiles": source_files,
            "sourceFilesSha256": source_files_sha256,
            "platform": target_tag,
        }
        kernel_inputs = receipt.get("kernel_inputs")
        if isinstance(kernel_inputs, dict) and isinstance(kernel_inputs.get("sha256"), str):
            manifest_payload["kernelInputsSha256"] = kernel_inputs["sha256"]

        manifest_path.write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{target_tag}"
        force_include = build_data.setdefault("force_include", {})
        force_include[str(binary.resolve())] = NATIVE_MEMBER
        force_include[str(manifest_path)] = "hillclimb/native/manifest.json"
        for notice_name in NOTICE_FILES:
            for candidate in (repo / notice_name, repo / "docs" / notice_name):
                if candidate.is_file():
                    force_include[str(candidate)] = f"hillclimb/notices/{notice_name}"
                    break
