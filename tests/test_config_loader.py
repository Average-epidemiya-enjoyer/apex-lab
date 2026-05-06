"""Tests for YAML config loading."""

from pathlib import Path

import pytest

from apex_lab.configs.loader import load_config


def test_load_default_config() -> None:
    config = load_config(Path(__file__).parent.parent / "experiments" / "default.yaml")
    assert config.experiment_name == "default"
    assert config.track == "oval"
    assert len(config.algorithms) == 2
    assert config.algorithms[0].name == "a_star"
    assert config.seed == 42


def test_load_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")


def test_load_config_missing_required_field(tmp_path: Path) -> None:
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("track: oval\nalgorithms:\n  - name: a_star\n")
    with pytest.raises(ValueError, match="experiment_name"):
        load_config(bad_yaml)
