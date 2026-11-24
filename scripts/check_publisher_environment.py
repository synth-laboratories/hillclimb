#!/usr/bin/env python3
"""Refuse publication if the actual GitHub environment lost its protection."""
from __future__ import annotations

import json
import subprocess
from typing import Any

from verify_publish import REPOSITORY, require

OWNER_USER_ID = 32282839


def check(environment: dict[str, Any], branches: dict[str, Any]) -> None:
    """Assert admin bypass prevention, owner review, and main-only branch policy."""
    assert isinstance(environment, dict)
    assert isinstance(branches, dict)
    require(
        environment.get("name") == "pypi" and environment.get("can_admins_bypass") is False,
        "publisher environment must prevent admin bypass",
    )
    rules = [rule for rule in environment.get("protection_rules", []) if rule.get("type") == "required_reviewers"]
    require(
        len(rules) == 1
        and len(rules[0].get("reviewers", [])) == 1
        and rules[0]["reviewers"][0].get("type") == "User"
        and rules[0]["reviewers"][0]["reviewer"].get("id") == OWNER_USER_ID,
        "owner-only environment review required",
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
    print("Protected pypi environment verified")
