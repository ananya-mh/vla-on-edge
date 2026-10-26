"""Tests for run metadata."""

from vla_on_edge.utils.run_metadata import git_info


def test_git_info_prefers_the_host_commit_from_the_environment(monkeypatch):
    monkeypatch.setenv("VLA_ON_EDGE_GIT_COMMIT", "abc123")
    monkeypatch.setenv("VLA_ON_EDGE_GIT_DIRTY", "true")
    assert git_info() == {"commit": "abc123", "dirty": True, "source": "host environment"}


def test_git_info_falls_back_to_git(monkeypatch):
    monkeypatch.delenv("VLA_ON_EDGE_GIT_COMMIT", raising=False)
    info = git_info()
    assert info["source"] == "git"
    assert info["commit"] and len(info["commit"]) == 40
