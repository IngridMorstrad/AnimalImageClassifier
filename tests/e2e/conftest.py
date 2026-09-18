"""Shared fixtures for the end-to-end test suite.

Only e2e tests live here: each test drives a real entry point (CLI or GUI app) and
asserts observable outcomes. Fixtures are added alongside the features they exercise.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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



# --------------------------------------------------------------------------- #
# The fixture card, and one shared `classify` run over it
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import make_e2e_fixtures  # noqa: E402 - needs the sys.path line above


@pytest.fixture(scope="session")
def fixture_card(tmp_path_factory) -> Path:
    """The synthetic SD-card tree from ``scripts/make_e2e_fixtures.py``.

    Built rather than committed: the sidecars encode exact box geometry, and a
    generated card keeps that geometry beside the expectations it justifies.
    """
    card = tmp_path_factory.mktemp("card")
    make_e2e_fixtures.build_card(card)
    return card


@pytest.fixture(scope="session")
def card_snapshot(fixture_card: Path) -> dict[str, str]:
    """``{relative path: sha256}`` for every file on the card, before any run.

    Invariant I1's evidence: the suite compares this against the same map taken
    after ``classify`` has run.
    """
    return _hash_tree(fixture_card)


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@dataclass(frozen=True)
class ClassifyRun:
    """One completed ``classify`` run, and cheap accessors into its results."""

    card: Path
    output: Path
    returncode: int
    stdout: str
    stderr: str

    @property
    def catalog(self) -> Path:
        return self.output / ".catalog.db"

    def tree(self) -> set[str]:
        """Every filed image as ``"<label>/<name>"`` — the output tree, exactly."""
        return {
            f"{path.parent.name}/{path.name}"
            for path in self.output.rglob("*")
            if path.is_file() and not path.name.startswith(".catalog.db")
        }

    def label_of(self, filename: str) -> str | None:
        """The label directory ``filename`` was filed into, or ``None``."""
        for entry in self.tree():
            label, name = entry.split("/", 1)
            if name == filename:
                return label
        return None

    def rows(self, sql: str, *params: object) -> list[sqlite3.Row]:
        """Query the catalog read-only, so a test can never mutate the run."""
        connection = sqlite3.connect(f"file:{self.catalog}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            return list(connection.execute(sql, params))
        finally:
            connection.close()

    def image_row(self, filename: str) -> sqlite3.Row:
        """The ``images`` row for a source file name."""
        found = self.rows(
            "SELECT i.* FROM images i JOIN sources s USING(sha256) "
            "WHERE s.path LIKE ?",
            f"%/{filename}",
        )
        assert len(found) == 1, f"expected one images row for {filename}, got {found}"
        return found[0]

    def boxes_of(self, filename: str) -> list[sqlite3.Row]:
        """The ``boxes`` rows for a source file name, in detector order."""
        return self.rows(
            "SELECT b.* FROM boxes b JOIN sources s USING(sha256) "
            "WHERE s.path LIKE ? ORDER BY b.idx",
            f"%/{filename}",
        )


@pytest.fixture(scope="session")
def classified(run_cli, fixture_card: Path, tmp_path_factory) -> ClassifyRun:
    """One real ``classify`` run over the fixture card, shared by the E-tests.

    Session-scoped because every assertion below is about the *same* run: they are
    different questions about one observable outcome, not different scenarios.
    """
    output = tmp_path_factory.mktemp("pics")
    result = run_cli(
        [
            "classify",
            str(fixture_card),
            "--output",
            str(output),
            "--detector",
            "scripted",
        ]
    )
    return ClassifyRun(
        card=fixture_card,
        output=output,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
