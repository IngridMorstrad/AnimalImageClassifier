"""E6, CLI-surface leg: the tool exposes no box-area floor, under any name.

E6's full form (chunk 9) drives a real ``classify`` run over the fixture card and
asserts the dominance rule decides alone. This leg is the part that can be proven
the moment a CLI exists, and it is the executable guard on invariant I2: if anyone
adds a ``--min-box-area``-style option, this fails immediately instead of being
caught by a code review six chunks later. ``dominance_ratio`` is the only size gate.
"""

from __future__ import annotations

import pytest

#: Every spelling a reintroduced absolute area floor would plausibly use. The
#: hyphenated forms are how typer renders an option; the underscored forms are how
#: a config key would read.
FLOOR_SPELLINGS = (
    "min-box-area",
    "min_box_area",
    "min-area",
    "min_area",
    "area-floor",
    "area_floor",
    "min-animal-area",
    "min_animal_area",
    "min-box-frac",
    "min_box_frac",
)

COMMANDS = ("classify", "gui", "train", "eval", "export-trainset", "verify")


def test_top_level_help_lists_every_command(run_cli) -> None:
    result = run_cli(["--help"])
    assert result.returncode == 0, result.stderr
    for command in COMMANDS:
        assert command in result.stdout, f"{command} missing from --help"


@pytest.mark.parametrize("command", COMMANDS)
def test_no_command_offers_an_area_floor(run_cli, command: str) -> None:
    """Invariant I2, per command: no option resembling an absolute area floor."""
    result = run_cli([command, "--help"])
    assert result.returncode == 0, result.stderr
    haystack = result.stdout.lower()
    offenders = [name for name in FLOOR_SPELLINGS if name in haystack]
    assert not offenders, (
        f"`{command} --help` offers {offenders}; dominance_ratio is the only size "
        "gate (DESIGN.md invariant I2). A distant animal is handled by the ratio, "
        "never by discarding small boxes."
    )
