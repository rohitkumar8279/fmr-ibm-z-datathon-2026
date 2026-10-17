"""Commit and push generated result artifacts to the FMR submission repository.

Authentication is read from GH_TOKEN or GITHUB_TOKEN at runtime. Git receives
an in-memory HTTP extra header; the token is never written into the remote URL
or repository configuration.
"""
from __future__ import annotations

import base64
import os
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/rohitkumar8279/fmr-ibm-z-datathon-2026.git"
DEFAULT_BRANCH = "main"


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, env=env).stdout.strip()


def _git_auth_env(token: str) -> dict[str, str]:
    env = os.environ.copy()
    encoded = base64.b64encode(("x-access-token:" + token).encode("utf-8")).decode("ascii")
    env.update({
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
        "GIT_CONFIG_VALUE_0": "AUTHORIZATION: basic " + encoded,
    })
    return env


def push(paths: str | list[str], message: str, branch: str = DEFAULT_BRANCH) -> None:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("Set GH_TOKEN or GITHUB_TOKEN in the notebook secret manager before pushing.")
    paths = [paths] if isinstance(paths, str) else list(paths)

    _run(["git", "config", "user.email", "fmr-colab-runner@users.noreply.github.com"])
    _run(["git", "config", "user.name", "FMR Colab Runner"])
    _run(["git", "remote", "set-url", "origin", REPO_URL])

    extra = [p for p in ("fmr/dashboard/data.js",) if Path(p).exists()]
    _run(["git", "add", *paths, *extra])
    status = _run(["git", "status", "--porcelain"])
    if not status:
        print("[push] nothing to commit")
        return
    _run(["git", "commit", "-m", message])

    env = _git_auth_env(token)
    _run(["git", "pull", "--rebase", "origin", branch], env=env)
    _run(["git", "push", "origin", f"HEAD:{branch}"], env=env)
    print(f"[push] pushed to {branch}: {message}")


if __name__ == "__main__":
    import sys

    push(sys.argv[1] if len(sys.argv) > 1 else "fmr/outputs/real",
         message=sys.argv[2] if len(sys.argv) > 2 else "Update real-model results")
