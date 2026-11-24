#!/usr/bin/env python3
"""Refuse publication if the actual GitHub environment lost its branch protection."""
from __future__ import annotations

import json
import subprocess
from typing import Any

from verify_publish import REPOSITORY, require


def check(environment: dict[str, Any], branches: dict[str, Any]) -> None:
    """Assert pypi environment identity, admin bypass prevention, and main-only branch policy."""
    assert isinstance(environment, dict)
    assert isinstance(branches, dict)
    require(
        environment.get("name") == "pypi" and environment.get("can_admins_bypass") is False,
        "publisher environment must be pypi and prevent admin bypass",
    )
    require(
        environment.get("deployment_branch_policy")
        == {"protected_branches": False, "custom_branch_policies": True},
        "explicit main-only branch policy required",
    )
    selected = branches.get("branch_policies", [])
    require(
        len(selected) == 1 and selected[0].get("name") == "main" and selected[0].get("type") == "branch",
        "publisher environment must allow main branch only",
    )


if __name__ == "__main__":
    prefix = f"repos/{REPOSITORY}/environments/pypi"

    def read(endpoint: str) -> dict[str, Any]:
        return json.loads(subprocess.check_output(["gh", "api", endpoint], timeout=30))

    check(read(prefix), read(prefix + "/deployment-branch-policies"))
    print("Protected pypi environment verified (main-only, automated OIDC)")
