from pathlib import Path

from artifact_paths import portable_artifact_path


def test_portable_artifact_path_relativizes_paths_inside_repo(tmp_path: Path):
    root = tmp_path / "repo"
    target = root / "artifacts" / "result.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")

    assert portable_artifact_path(target, repo_root=root) == "artifacts/result.json"


def test_portable_artifact_path_keeps_external_path_absolute(tmp_path: Path):
    root = tmp_path / "repo"
    outside = tmp_path / "external" / "result.json"

    assert portable_artifact_path(outside, repo_root=root) == str(outside.resolve())


def test_portable_artifact_path_preserves_none():
    assert portable_artifact_path(None) is None
