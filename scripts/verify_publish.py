#!/usr/bin/env python3
"""Validate selected release bytes before granting the publisher any work."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import time
import tomllib
from typing import Optional
import urllib.error
import urllib.request
import zipfile

PACKAGE_NAME = "synth-hillclimb"
WHEEL_PREFIX = "synth_hillclimb"
REPOSITORY = "synth-laboratories/hillclimb"
MANIFEST_SCHEMA = "synth-hillclimb.publish-manifest.v1"
COHORT_SCHEMA = "synth-hillclimb.release-cohort.v1"
WHEEL_NAME_RE = re.compile(
    r"^synth_hillclimb-(?P<version>[A-Za-z0-9_.]+)-(?P<py>[A-Za-z0-9_.]+)-(?P<abi>[A-Za-z0-9_]+)-(?P<plat>[A-Za-z0-9_.]+)\.whl$"
)


def require(condition: bool, message: str) -> None:
    """Raise a deterministic validation error when an invariant fails."""
    if not condition:
        raise ValueError(message)


def sha(data: bytes) -> str:
    """Return lowercase hexadecimal SHA-256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def request(tag: str, channel: str, manifest_sha256: str) -> str:
    """Validate dispatch inputs and return the normalized PEP 440 version."""
    require(channel in {"stable", "nightly"}, "invalid channel")
    require(len(tag) <= 65, "release tag exceeds bound")
    version = tag.removeprefix("v")
    pattern = r"\d+\.\d+\.\d+" + (r"\.dev\d+" if channel == "nightly" else "")
    require(tag == "v" + version and re.fullmatch(pattern, version) is not None, "channel/version mismatch")
    require(re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is not None, "invalid manifest digest")
    assert version
    assert len(manifest_sha256) == 64
    return version


def bounded_file(path: Path, limit: int) -> bytes:
    """Read a non-symlink regular file up to a strict byte bound."""
    assert limit > 0
    require(not path.is_symlink() and path.is_file(), "release asset must be a regular file")
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, "release asset exceeds bound")
    return data


def _verify_source_tree(
    source: str, tag: str, channel: str, files: dict[str, str], version: str
) -> None:
    """Check tag commit, branch ancestry, exact source hashes, and project version."""
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], timeout=15)

    require(git("rev-parse", tag + "^{commit}").decode().strip() == source, "release tag source mismatch")
    branch = "nightly" if channel == "nightly" else "main"
    subprocess.run(["git", "merge-base", "--is-ancestor", source, "origin/" + branch], check=True, timeout=15)
    raw_names = git("ls-tree", "-r", "--name-only", "-z", source).decode().rstrip("\0")
    names = raw_names.split("\0") if raw_names else []
    require(set(names) == set(files), "source tree file list mismatch")
    for name, digest in files.items():
        require(sha(git("show", source + ":" + name)) == digest, "source tree hash mismatch")
    matching_versions: list[str] = []
    for name in names:
        if PurePosixPath(name).name != "pyproject.toml":
            continue
        project = tomllib.loads(git("show", source + ":" + name).decode())
        if project.get("project", {}).get("name") == PACKAGE_NAME:
            matching_versions.append(project["project"]["version"])
    require(len(matching_versions) == 1, "expected exactly one synth-hillclimb pyproject.toml")
    require(matching_versions[0] == version, "source version mismatch")


def verify(
    assets: Path,
    tag: str,
    channel: str,
    manifest_sha256: str,
    check_git: bool = False,
) -> dict[str, str]:
    """Verify release manifest, cohort, wheel integrity, and optional git bindings."""
    version = request(tag, channel, manifest_sha256)
    raw = bounded_file(assets / "publish-manifest.json", 16_384)
    require(sha(raw) == manifest_sha256, "publication manifest digest mismatch")
    manifest = json.loads(raw)
    require(
        set(manifest)
        == {
            "schema",
            "channel",
            "version",
            "wheel",
            "cohort_sha256",
            "source_commit",
            "source_files_sha256",
            "native_sha256",
        },
        "unexpected publication fields",
    )
    require(manifest["schema"] == MANIFEST_SCHEMA, "unsupported publication schema")
    require(manifest["version"] == version and manifest["channel"] == channel, "publication selection mismatch")
    require(set(manifest["wheel"]) == {"filename", "sha256"}, "unexpected wheel fields")
    for value in [
        manifest["cohort_sha256"],
        manifest["source_files_sha256"],
        manifest["native_sha256"],
        manifest["wheel"]["sha256"],
    ]:
        require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, "invalid asset digest")
    require(
        isinstance(manifest["source_commit"], str)
        and re.fullmatch(r"[0-9a-f]{40}", manifest["source_commit"]) is not None,
        "invalid source commit",
    )
    filename = manifest["wheel"]["filename"]
    require(
        isinstance(filename, str) and PurePosixPath(filename).name == filename,
        "unexpected wheel filename/platform",
    )
    wheel_match = WHEEL_NAME_RE.fullmatch(filename)
    require(
        wheel_match is not None and wheel_match.group("version") == version,
        "unexpected wheel filename/platform",
    )
    require({p.name for p in assets.glob("*.whl")} == {filename}, "unselected wheels present")

    cohort_raw = bounded_file(assets / "cohort.json", 65_536)
    cohort = json.loads(cohort_raw)
    canonical = json.dumps(cohort, sort_keys=True, separators=(",", ":")).encode()
    require(cohort_raw == canonical + b"\n", "cohort must use canonical JSON")
    require(sha(canonical) == manifest["cohort_sha256"], "cohort digest mismatch")
    require(cohort["version"] == version and cohort["channel"] == channel, "cohort selection mismatch")
    require(cohort["schema"] == COHORT_SCHEMA, "unsupported cohort schema")
    require(cohort["source"]["qualified_commit"] == manifest["source_commit"], "cohort source mismatch")
    require(cohort["source"]["source_files_sha256"] == manifest["source_files_sha256"], "cohort source digest mismatch")
    require(
        cohort["artifacts"]["wheel"]["filename"] == filename
        and cohort["artifacts"]["wheel"]["sha256"] == manifest["wheel"]["sha256"],
        "cohort wheel mismatch",
    )
    require(cohort["artifacts"]["native"]["sha256"] == manifest["native_sha256"], "cohort native mismatch")

    wheel = assets / filename
    require(sha(bounded_file(wheel, 64 * 1024**2)) == manifest["wheel"]["sha256"], "wheel digest mismatch")
    with zipfile.ZipFile(wheel) as archive:
        entries = archive.infolist()
        require(
            len(entries) <= 512 and len({e.filename for e in entries}) == len(entries),
            "ambiguous wheel members",
        )
        for entry in entries:
            path = PurePosixPath(entry.filename)
            require(
                not path.is_absolute() and ".." not in path.parts and "\\" not in entry.filename,
                "unsafe wheel member",
            )
            require(not stat.S_ISLNK(entry.external_attr >> 16), "wheel symlink refused")
            require(entry.file_size <= 32 * 1024**2, "wheel member exceeds bound")
        metadata = BytesParser().parsebytes(archive.read(f"{WHEEL_PREFIX}-{version}.dist-info/METADATA"))
        require(metadata["Name"] == PACKAGE_NAME and metadata["Version"] == version, "wheel metadata mismatch")
        require(metadata["License-Expression"] == "Apache-2.0", "wheel license mismatch")

        native_manifest = json.loads(archive.read("hillclimb/native/manifest.json"))
        require(
            native_manifest["version"] == version and native_manifest["sourceDirty"] is False,
            "unqualified native source",
        )
        require(native_manifest["sourceCommit"] == manifest["source_commit"], "native source commit mismatch")
        require(
            native_manifest["sourceFilesSha256"] == manifest["source_files_sha256"],
            "native source cohort mismatch",
        )
        require(native_manifest["sha256"] == manifest["native_sha256"], "native manifest digest mismatch")
        native_member = native_manifest.get("nativeMember")
        require(
            isinstance(native_member, str)
            and native_member.startswith("hillclimb/")
            and ".." not in PurePosixPath(native_member).parts
            and "\\" not in native_member,
            "invalid native member path",
        )
        require(sha(archive.read(native_member)) == manifest["native_sha256"], "native bytes mismatch")
        files = native_manifest["sourceFiles"]
        require(isinstance(files, dict) and len(files) <= 4096, "source manifest exceeds bound")
        require(
            sha(json.dumps(files, sort_keys=True, separators=(",", ":")).encode())
            == manifest["source_files_sha256"],
            "source manifest digest mismatch",
        )
        for name, digest in files.items():
            path = PurePosixPath(name)
            require(
                not path.is_absolute()
                and ".." not in path.parts
                and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                "invalid source manifest entry",
            )

    if check_git:
        _verify_source_tree(manifest["source_commit"], tag, channel, files, version)

    result = {
        "filename": filename,
        "version": version,
        "source_commit": manifest["source_commit"],
        "wheel_sha256": manifest["wheel"]["sha256"],
        "cohort_sha256": manifest["cohort_sha256"],
    }
    assert result["wheel_sha256"] == manifest["wheel"]["sha256"]
    assert result["version"] == version
    return result


def verify_registry(result: dict[str, str]) -> None:
    """Confirm the published PyPI JSON API reports the exact uploaded wheel digest."""
    url = f"https://pypi.org/pypi/{PACKAGE_NAME}/{result['version']}/json"
    for attempt in range(10):
        try:
            with urllib.request.urlopen(url, timeout=15) as response:
                observed = json.loads(response.read(1024 * 1024 + 1))
            files = observed["urls"]
            require(
                len(files) == 1
                and files[0]["filename"] == result["filename"]
                and files[0]["digests"]["sha256"] == result["wheel_sha256"],
                "registry wheel mismatch",
            )
            require(
                observed["info"]["name"] == PACKAGE_NAME and observed["info"]["version"] == result["version"],
                "registry metadata mismatch",
            )
            return
        except urllib.error.HTTPError as error:
            if error.code != 404 or attempt == 9:
                raise
            time.sleep(3)


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entrypoint for pre-publish and post-publish release verification."""
    parser = argparse.ArgumentParser(description="Verify synth-hillclimb release manifests and wheels.")
    parser.add_argument("--assets", type=Path, default=Path("artifacts/publish"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--channel", choices=["stable", "nightly"], required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--request-only", action="store_true")
    parser.add_argument("--check-git", action="store_true")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--registry", action="store_true")
    args = parser.parse_args(argv)
    request(args.tag, args.channel, args.manifest_sha256)
    if args.request_only:
        return
    result = verify(args.assets, args.tag, args.channel, args.manifest_sha256, args.check_git)
    if args.registry:
        verify_registry(result)
    if args.github_output:
        with args.github_output.open("a") as stream:
            for key, value in result.items():
                stream.write(f"{key}={value}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
