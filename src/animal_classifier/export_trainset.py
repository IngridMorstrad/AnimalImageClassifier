"""``export-trainset``: catalog → training manifest (DESIGN.md §7.5).

Closes the loop: GUI corrections (and, opt-in, high-confidence model labels) become
the next training set. Reads the catalog read-only and the source images not at all
in the default path — the ``box`` comes from the stored dominant box.

**The label filter is explicit, and its counts are printed**, because most human
overrides are not species (``junk``, ``landscape``, …) and emitting ``junk`` into a
species manifest would train the head on a non-species class:

| Catalog label | Default | ``--include-non-species`` |
|---|---|---|
| a species slug | exported (with the dominant box) | exported |
| ``landscape`` | skipped, counted | exported as class ``landscape`` |
| ``junk`` | skipped, counted | exported as class ``junk`` |
| ``multiple`` | skipped, counted | still skipped (no single box owns it) |
| ``unknown`` | skipped, counted | still skipped (records absence of identity) |

Every run prints ``exported=… skipped_multiple=… skipped_unknown=…
skipped_landscape=… skipped_junk=… no_box=…`` so a user who re-tagged 200 images and
got 40 lines can see where the other 160 went.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from .config import Config
from .decide import LABEL_JUNK, LABEL_LANDSCAPE, LABEL_MULTIPLE, LABEL_UNKNOWN
from .errors import CatalogError

log = logging.getLogger(__name__)

_ALWAYS_SKIP = frozenset({LABEL_MULTIPLE, LABEL_UNKNOWN})
_SCENE_CLASSES = frozenset({LABEL_LANDSCAPE, LABEL_JUNK})


def run_export(
    config: Config,
    *,
    destination: Path,
    include_model_labels: bool,
    min_conf: float,
    include_non_species: bool,
) -> str:
    """Walk the catalog and write the manifest. Returns the counted summary."""
    if not config.catalog_path.is_file():
        raise CatalogError(
            f"no catalog at {config.catalog_path}; run `classify` first "
            "or point --output at the root you filed into"
        )

    counts = {
        "exported": 0,
        "skipped_multiple": 0,
        "skipped_unknown": 0,
        "skipped_landscape": 0,
        "skipped_junk": 0,
        "no_box": 0,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(f"file:{config.catalog_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT sha256, label, label_source, confidence, dest_path "
            "FROM images WHERE status='done' AND label IS NOT NULL "
            "ORDER BY sha256"
        ).fetchall()
        with destination.open("w", encoding="utf-8") as handle:
            for row in rows:
                line = _emit_line(
                    conn, row, counts,
                    include_model_labels=include_model_labels,
                    min_conf=min_conf,
                    include_non_species=include_non_species,
                )
                if line is not None:
                    handle.write(line + "\n")
    finally:
        conn.close()

    summary = " ".join(f"{k}={v}" for k, v in counts.items())
    log.info("export-trainset: %s -> %s", summary, destination)
    return summary


def _emit_line(conn, row, counts, *, include_model_labels, min_conf, include_non_species):
    label = row["label"]
    is_human = row["label_source"] == "human"

    # multiple/unknown are always skipped and counted, whatever their source: no
    # single box owns a `multiple`, and `unknown` records absence of an identity.
    if label == LABEL_MULTIPLE:
        counts["skipped_multiple"] += 1
        return None
    if label == LABEL_UNKNOWN:
        counts["skipped_unknown"] += 1
        return None

    # Scene classes (landscape/junk) are reserved, so they are only ever *model*
    # labels and carry no confidence — they are judged by --include-non-species,
    # not by the confidence gate below (a NULL confidence must not hide them).
    if label in _SCENE_CLASSES:
        if not include_non_species:
            counts["skipped_landscape" if label == LABEL_LANDSCAPE else "skipped_junk"] += 1
            return None
        counts["exported"] += 1
        return json.dumps({"path": row["dest_path"], "label": label})

    # A species slug. Human labels always export; model labels only with the opt-in
    # and above the confidence floor.
    if not is_human:
        if not include_model_labels:
            return None
        if row["confidence"] is None or row["confidence"] < min_conf:
            return None

    # A species slug: emit with the image's dominant box. A human-labelled *herd*
    # frame has no dominant box by construction (that is what made it `multiple`),
    # so fall back to the largest animal box: on a single-species herd it is a
    # correct, well-framed crop of that species. Without the fallback these samples
    # trained on the whole frame instead — mostly grass.
    box = conn.execute(
        "SELECT x0, y0, x1, y1 FROM boxes WHERE sha256=? AND is_dominant=1 LIMIT 1",
        (row["sha256"],),
    ).fetchone()
    if box is None:
        box = conn.execute(
            "SELECT x0, y0, x1, y1 FROM boxes WHERE sha256=? AND cls='animal' "
            "ORDER BY area_frac DESC LIMIT 1",
            (row["sha256"],),
        ).fetchone()
    record = {"path": row["dest_path"], "label": label}
    if box is not None:
        record["box"] = [box["x0"], box["y0"], box["x1"], box["y1"]]
    else:
        # Reachable only via a human override on a box-less image (§7.5).
        counts["no_box"] += 1
    counts["exported"] += 1
    return json.dumps(record)
