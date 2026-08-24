# -*- coding: utf-8 -*-
"""Portable path labels for persisted experiment metadata."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]
REPO_ROOT = Path(__file__).resolve().parents[1]


def portable_artifact_path(
    path: Optional[PathLike], *, repo_root: Optional[PathLike] = None
) -> Optional[str]:
    """Return a repo-relative POSIX path when possible, else an absolute path.

    Runtime code may still use resolved ``Path`` objects. This helper is only for
    values persisted to JSON/CSV manifests, so a cloned repository does not retain
    the producing machine's user directory.
    """
    if path is None:
        return None
    resolved = Path(path).resolve()
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)
