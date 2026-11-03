from __future__ import annotations

import os

import pytest

from demand import cli
from demand.config import Config


def test_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_main_dispatches_the_selected_command(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(cli, "load_config", lambda *paths: object())
    monkeypatch.setattr(cli, "cmd_synth", lambda cfg: seen.append("synth") or 0)

    assert cli.main(["synth"]) == 0
    assert seen == ["synth"]


def test_config_overlays_are_forwarded_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    paths: list[str] = []
    monkeypatch.setattr(cli, "load_config", lambda *items: paths.extend(items) or object())
    monkeypatch.setattr(cli, "cmd_report", lambda cfg: 0)

    cli.main(["--config", "a.yaml", "--config", "b.yaml", "report"])
    assert paths == ["a.yaml", "b.yaml"]


def test_backtest_flag_reaches_the_command(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []
    monkeypatch.setattr(cli, "load_config", lambda *paths: object())
    monkeypatch.setattr(
        cli, "cmd_backtest", lambda cfg, with_prophet=False: seen.append(with_prophet) or 0
    )

    cli.main(["backtest", "--with-prophet"])
    assert seen == [True]


def test_synth_writes_the_expected_files(demo_config: Config) -> None:
    assert cli.cmd_synth(demo_config) == 0
    assert demo_config.data.sales_path.exists()
    assert demo_config.data.calendar_path.exists()
    assert demo_config.data.prices_path.exists()


def test_serve_sets_the_artifact_env_and_worker_count(
    demo_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    seen: dict[str, object] = {}
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: seen.update(kwargs))

    assert cli.cmd_serve(demo_config, workers=2) == 0
    assert seen["workers"] == 2
    assert seen["host"] == demo_config.serve.host
    assert os.environ["DEMAND_ARTIFACTS"] == str(demo_config.artifacts_dir.resolve())


def test_dashboard_command_uses_the_next_port(
    demo_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_call(command: list[str], env: dict[str, str]) -> int:
        seen["command"] = command
        seen["env"] = env
        return 0

    monkeypatch.setattr(cli.subprocess, "call", fake_call)

    assert cli.cmd_dashboard(demo_config) == 0
    command = seen["command"]
    assert isinstance(command, list)
    assert "--server.address" in command
    assert str(demo_config.serve.port + 1) in command
    assert seen["env"]["DEMAND_ARTIFACTS"] == str(demo_config.artifacts_dir.resolve())


def test_demo_stops_at_the_first_failing_step(
    demo_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "cmd_synth", lambda cfg: 2)
    assert cli.cmd_demo(demo_config) == 2
