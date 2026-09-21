"""The systemd units run commands the CLI actually accepts.

``ssd-cleaner.service`` ran ``python -m animaltracker.cli cleanup --config
...``. ``--config`` is an option of the top-level parser, so after the
subcommand argparse rejects it ("unrecognized arguments") and the unit
failed on every run. Nobody noticed because it had never been installed.
Each unit's ExecStart is parsed here with the real parser.
"""
from __future__ import annotations

import configparser
import shlex
from pathlib import Path

import pytest

from animaltracker.cli import build_parser, cmd_cleanup, cmd_run

UNITS = Path(__file__).resolve().parents[1] / "systemd"


def unit(name: str) -> configparser.RawConfigParser:
    text = (UNITS / name).read_text().replace("\\\n", " ")    # systemd line continuations
    parser = configparser.RawConfigParser(strict=False)
    parser.optionxform = str                                   # keys are case-sensitive
    parser.read_string(text)
    return parser


def cli_args(name: str) -> list:
    """What follows ``-m animaltracker`` / ``-m animaltracker.cli`` in ExecStart."""
    argv = shlex.split(unit(name)["Service"]["ExecStart"])
    module = argv.index("-m") + 1
    assert argv[module] in ("animaltracker", "animaltracker.cli"), argv
    return argv[module + 1:]


@pytest.mark.parametrize("name, command, func", [
    ("animaltracker.service", "run", cmd_run),
    ("ssd-cleaner.service", "cleanup", cmd_cleanup),
])
def test_each_unit_runs_a_command_the_cli_accepts(name, command, func):
    args = build_parser().parse_args(cli_args(name))    # SystemExit(2) is the old bug

    assert args.command == command
    assert args.func is func
    assert args.config.endswith("/config/cameras.yml"), "the unit must name its config"


def test_the_scheduled_pass_is_a_real_one_not_a_preview():
    args = build_parser().parse_args(cli_args("ssd-cleaner.service"))

    assert args.dry_run is False


def test_the_cleanup_uses_the_same_install_as_the_pipeline():
    service = unit("animaltracker.service")["Service"]
    cleaner = unit("ssd-cleaner.service")["Service"]

    for key in ("User", "WorkingDirectory", "EnvironmentFile"):
        assert cleaner[key] == service[key], key
    assert cli_args("ssd-cleaner.service")[1] == cli_args("animaltracker.service")[1]


def test_the_timer_is_daily_and_catches_up_after_downtime():
    timer = unit("ssd-cleaner.timer")["Timer"]

    assert timer["OnCalendar"] == "daily"
    assert timer["Persistent"] == "true"
    assert timer["Unit"] == "ssd-cleaner.service"
