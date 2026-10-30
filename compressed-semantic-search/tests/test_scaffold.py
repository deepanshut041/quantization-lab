"""Check configuration identity and honest command boundaries."""
import json
from importlib import resources

import pytest

from core.config import load_config
from core.workflow import PENDING_ACTIONS
from utils.commands import main


def test_status_does_not_claim_model_execution(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["check"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["project"] == "compressed-semantic-search"
    assert report["model"]["repo_id"] == "google/embeddinggemma-2"
    assert report["model_execution_verified"] is False
    assert report["status"] == "boilerplate"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("action", PENDING_ACTIONS)
def test_future_actions_fail_without_creating_outputs(action, capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main([action]) == 2
    result = json.loads(capsys.readouterr().err)
    assert result["action"] == action
    assert result["status"] == "not_implemented"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("replacement", ["main", "v1", "0" * 39])
def test_mutable_or_invalid_model_revision_is_rejected(replacement, tmp_path):
    text = resources.files("core").joinpath("project.toml").read_text()
    config = load_config()
    path = tmp_path / "invalid.toml"
    path.write_text(text.replace(config["model"]["revision"], replacement))
    with pytest.raises(ValueError, match="exact 40-character"):
        load_config(path)


def test_invalid_budget_does_not_become_a_run(capsys, tmp_path):
    text = resources.files("core").joinpath("project.toml").read_text()
    path = tmp_path / "invalid.toml"
    path.write_text(text.replace("embedding_dimensions = 768",
                                 "embedding_dimensions = 0"))
    assert main(["prepare-data", "--config", str(path)]) == 1
    assert json.loads(capsys.readouterr().err)["status"] == "error"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["invalid.toml"]
