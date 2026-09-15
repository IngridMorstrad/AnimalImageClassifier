"""Shared fixtures for the end-to-end test suite.

Only e2e tests live here: each test drives a real entry point (CLI or GUI app) and
asserts observable outcomes. Fixtures are added alongside the features they exercise.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence

import pytest


@pytest.fixture(scope="session")
def cli_path() -> str:
    """Absolute path to the installed ``animal-classifier`` console script.

    Resolved from ``PATH`` rather than constructed, so the tests exercise the same
    entry point a user runs. A missing script fails loudly here instead of being
    silently replaced by ``python -m``, which would test a different surface.
    """
    found = shutil.which("animal-classifier")
    if found is None:
        raise AssertionError(
            "the animal-classifier console script is not on PATH; run the suite as "
            "`uv run --frozen pytest tests/e2e -q` so the project venv is active"
        )
    return found


@pytest.fixture(scope="session")
def run_cli(cli_path: str):
    """Invoke the real CLI as a subprocess and return the completed process."""

    def _run(args: Sequence[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - fixed executable, no shell
            [cli_path, *args],
            capture_output=True,
            text=True,
            check=False,
            **kwargs,  # type: ignore[arg-type]
        )

    return _run
