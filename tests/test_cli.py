"""Tests for the CLI entry point."""

import pytest

from apex_lab.cli import build_parser, cmd_list_algorithms


def test_list_algorithms_runs(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    args = parser.parse_args(["list-algorithms"])
    args.func(args)
    captured = capsys.readouterr()
    assert "a_star" in captured.out
    assert "ppo" in captured.out


def test_run_requires_config() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run"])


def test_compare_requires_algorithms_and_track() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["compare", "--track", "oval"])
