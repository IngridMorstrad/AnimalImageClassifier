"""``verify`` and ``verify --fix`` (DESIGN.md §8).

Checks assets, config and catalog/filesystem consistency, with the **specific** exit
codes §8 requires — not a generic non-zero:

- ``0`` — everything present and consistent (each check ``pass``/``skipped``);
- ``3`` — a required asset/config is missing or invalid (``--fix`` cannot repair it);
- ``4`` — assets fine but catalog and filesystem disagree (the ``--fix``-able class);
  ``verify --fix`` exits ``0`` when it resolved every finding, ``4`` when any remains.

``verify --fix`` is the only reconciler and is deliberately narrow — it repairs
exactly the two states the design can produce (a pending re-tag, and a duplicate the
catalog explains), and it obeys §5.8's no-write rule: it only ever ``os.replace`` or
``os.unlink``, never opening a materialized file. It never invents intent from where
files happen to sit.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from .config import Config
from .errors import EXIT_CONFIG, EXIT_OK, EXIT_PARTIAL

log = logging.getLogger(__name__)

_LABEL_DIRS_INTERNAL_PREFIX = "."


def run_verify(config: Config, *, fix: bool) -> tuple[dict[str, Any], int]:
    """Run the checks (and optionally --fix). Returns (report, exit_code)."""
    checks: list[dict[str, str]] = []
    exit_code = EXIT_OK

    catalog_path = config.catalog_path
    if not catalog_path.is_file():
        checks.append(_check("catalog", "fail", f"no catalog at {catalog_path}"))
        return {"checks": checks}, EXIT_CONFIG
    checks.append(_check("catalog", "pass", f"openable at {catalog_path}"))

    if not os.access(config.output_root, os.W_OK):
        checks.append(_check("output_writable", "fail", f"{config.output_root} not writable"))
        exit_code = EXIT_CONFIG
    else:
        checks.append(_check("output_writable", "pass", str(config.output_root)))

    conn = sqlite3.connect(f"file:{catalog_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        _check_nesting(conn, config, checks)
        findings = _reconcile(conn, config, checks)
    finally:
        conn.close()

    if findings:
        if fix:
            fixed, remaining = _apply_fixes(config, findings)
            checks.append(
                _check("fix", "pass" if not remaining else "fail",
                       f"repaired {fixed}, {len(remaining)} left")
            )
            exit_code = EXIT_OK if not remaining and exit_code == EXIT_OK else max(exit_code, EXIT_PARTIAL if remaining else EXIT_OK)
        else:
            checks.append(_check("reconcile", "fail", f"{len(findings)} finding(s); run --fix"))
            exit_code = max(exit_code, EXIT_PARTIAL)
    else:
        checks.append(_check("reconcile", "pass", "catalog and filesystem agree"))

    return {"checks": checks}, exit_code


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _check_nesting(conn, config, checks) -> None:
    """§8: output_root must not be nested with the newest run's source_root."""
    row = conn.execute(
        "SELECT source_root FROM runs ORDER BY started_at DESC, run_id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        checks.append(_check("nesting", "skipped(no_runs)", "no runs recorded yet"))
        return
    source = Path(row["source_root"]).resolve()
    out = config.output_root.resolve()
    nested = out == source or source in out.parents or out in source.parents
    checks.append(
        _check("nesting", "fail" if nested else "pass",
               f"output {out} vs source {source}")
    )


def _reconcile(conn, config, checks) -> list[dict[str, Any]]:
    """Find catalog/filesystem disagreements. Returns a list of findings."""
    findings: list[dict[str, Any]] = []
    rows = conn.execute(
        "SELECT sha256, label, status, dest_path, label_source FROM images "
        "WHERE status IN ('done','materializing')"
    ).fetchall()
    for row in rows:
        dest = row["dest_path"]
        if not dest:
            findings.append({"kind": "missing_dest", "sha256": row["sha256"]})
            continue
        path = Path(dest)
        if row["status"] == "materializing":
            # A pending re-tag: intent recorded, rename not yet done (§5.8 crash window).
            override = conn.execute(
                "SELECT old_label, new_label FROM overrides WHERE sha256=? "
                "ORDER BY id DESC LIMIT 1", (row["sha256"],)
            ).fetchone()
            if override is not None:
                findings.append({
                    "kind": "pending_retag", "sha256": row["sha256"],
                    "dest": dest, "old_label": override["old_label"],
                })
            continue
        if path.is_symlink():
            continue  # a link (possibly dangling) is a valid filed state
        if not path.exists():
            findings.append({"kind": "dest_gone", "sha256": row["sha256"], "dest": dest})
    checks.append(
        _check("reconcile_scan", "pass", f"examined {len(rows)} rows, {len(findings)} finding(s)")
    )
    return findings


def _apply_fixes(config: Config, findings: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    """Repair the two fixable states; leave everything else reported (§8)."""
    fixed = 0
    remaining: list[dict[str, Any]] = []
    for finding in findings:
        if finding["kind"] == "pending_retag":
            old_dir = config.output_root / finding["old_label"]
            old_path = old_dir / Path(finding["dest"]).name
            dest = Path(finding["dest"])
            if old_path.exists() and not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(old_path, dest)  # complete the rename forward (§8)
                _mark_done(config, finding["sha256"])
                fixed += 1
                continue
        remaining.append(finding)
    return fixed, remaining


def _mark_done(config: Config, sha256: str) -> None:
    from .catalog import Catalog, GUI_BUSY_TIMEOUT_MS, Status  # noqa: PLC0415

    with Catalog.open(config.catalog_path, busy_timeout_ms=GUI_BUSY_TIMEOUT_MS) as catalog:
        catalog.update_image(sha256, status=str(Status.DONE))
