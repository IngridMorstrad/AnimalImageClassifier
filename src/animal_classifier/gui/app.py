"""The review GUI app (DESIGN.md §6).

``create_app(config)`` returns a FastAPI app with the §6 routes. The design points
worth restating, because they are where a GUI usually cuts a corner:

- **The catalog is authoritative for counts; the filesystem is a cross-check.**
  ``/api/labels`` returns ``count`` from the catalog and ``files_on_disk`` from the
  directory, so a ``--dry-run`` catalog (rows ``planned``, no directories) shows a
  populated grid beside ``files_on_disk == 0`` instead of an empty sidebar.
- **Unscored rows are never silently filtered.** ``images.confidence`` is NULL for
  ``landscape``/``junk``/``multiple``/box-less ``unknown``; a confidence filter
  matches those only with ``include_unscored=true``, and every response carries
  ``unscored_excluded`` so the UI can offer the toggle.
- **Byte routes read ``dest_path`` only, never the source and never a request
  path.** ``planned`` / missing / dangling-symlink destinations are a 409 with the
  reason, so a ``--link`` user whose card is unplugged gets a placeholder tile, not
  a crash.
- **Re-tag is a pure output-tree rename** via :func:`materialize.retag`, with the
  override recorded first (I4). Contention with a running ``classify`` writer is a
  fast 409, never a hung request.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

from ..catalog import GUI_BUSY_TIMEOUT_MS, Catalog, LabelSource, Status
from ..config import Config
from ..errors import CatalogLockedError, MaterializeError
from ..materialize import retag
from ..taxonomy import LABEL_RE, RESERVED_LABELS, merged

log = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_THUMB_EDGE = 320


def create_app(config: Config) -> Any:
    """Build the FastAPI app over ``config.output_root`` (§6)."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import (
        FileResponse,
        HTMLResponse,
        JSONResponse,
        Response,
    )

    app = FastAPI(title="animal-classifier review", openapi_url=None)
    output_root = config.output_root
    static_dir = Path(__file__).parent / "static"
    known_labels = _known_label_space(config)

    def _ro() -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{config.catalog_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (static_dir / "index.html").read_text(encoding="utf-8")

    @app.get("/static/{name}")
    def static(name: str) -> Any:
        candidate = static_dir / name
        if not candidate.is_file() or candidate.parent != static_dir:
            raise HTTPException(404, "not found")
        media = "text/css" if name.endswith(".css") else "application/javascript"
        return Response(candidate.read_text(encoding="utf-8"), media_type=media)

    @app.get("/api/labels")
    def api_labels() -> dict[str, Any]:
        with _ro() as conn:
            rows = conn.execute(
                "SELECT label, COUNT(*) AS n FROM images "
                "WHERE status IN ('done','planned') AND label IS NOT NULL "
                "GROUP BY label ORDER BY label"
            ).fetchall()
        labels = []
        for row in rows:
            directory = output_root / row["label"]
            files_on_disk = _count_files_on_disk(directory)
            labels.append(
                {"label": row["label"], "count": row["n"], "files_on_disk": files_on_disk}
            )
        return {"labels": labels}

    @app.get("/api/images")
    def api_images(
        label: str | None = None,
        min_conf: float | None = None,
        max_conf: float | None = None,
        include_unscored: bool = False,
        q: str | None = None,
        limit: int = 60,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit = max(1, min(limit, 200))
        clauses = ["status IN ('done','planned')"]
        params: list[Any] = []
        if label:
            clauses.append("label = ?")
            params.append(label)
        if q:
            clauses.append("(label LIKE ? OR species_common LIKE ?)")
            params += [f"%{q}%", f"%{q}%"]

        conf_active = min_conf is not None or max_conf is not None
        unscored_excluded = 0
        with _ro() as conn:
            if conf_active and not include_unscored:
                unscored_excluded = _count_unscored(conn, clauses, params)
            conf_clauses = list(clauses)
            if min_conf is not None:
                conf_clauses.append(
                    "(confidence >= ?" + (" OR confidence IS NULL)" if include_unscored else ")")
                )
                params.append(min_conf)
            if max_conf is not None:
                conf_clauses.append(
                    "(confidence <= ?" + (" OR confidence IS NULL)" if include_unscored else ")")
                )
                params.append(max_conf)
            where = " AND ".join(conf_clauses)
            total = conn.execute(
                f"SELECT COUNT(*) AS n FROM images WHERE {where}", params
            ).fetchone()["n"]
            rows = conn.execute(
                f"SELECT * FROM images WHERE {where} "
                "ORDER BY last_updated DESC, sha256 LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            items = [_image_payload(conn, row) for row in rows]
        return {"items": items, "total": total, "unscored_excluded": unscored_excluded}

    @app.get("/api/images/{sha256}/thumb")
    def api_thumb(sha256: str) -> Any:
        path = _readable_dest(sha256)
        thumb = _thumbnail(sha256, path, output_root)
        return FileResponse(thumb, media_type="image/jpeg")

    @app.get("/api/images/{sha256}/full")
    def api_full(sha256: str) -> Any:
        path = _readable_dest(sha256)
        return FileResponse(path)

    @app.post("/api/images/{sha256}/label")
    def api_label(sha256: str, body: dict[str, Any]) -> Any:
        _validate_sha(sha256)
        new_label = str(body.get("label", "")).strip()
        note = body.get("note")
        _validate_label(new_label, known_labels, config.allow_new_labels)

        with _ro() as conn:
            row = conn.execute("SELECT * FROM images WHERE sha256 = ?", (sha256,)).fetchone()
        if row is None:
            raise HTTPException(404, "unknown image")
        if not row["dest_path"]:
            raise HTTPException(409, {"error": "image has no materialized file", "reason": "missing"})

        try:
            new_path = _do_retag(config, sha256, row, new_label, note)
        except CatalogLockedError:
            return JSONResponse(
                {"error": "a classify run is writing the catalog; retry"}, status_code=409
            )
        except MaterializeError as error:
            raise HTTPException(409, str(error)) from error
        return {"dest_path": str(new_path), "label": new_label}

    @app.get("/api/review")
    def api_review(limit: int = 60, offset: int = 0) -> dict[str, Any]:
        """The active-learning queue: animals the model could not confidently name.

        An image qualifies when it has at least one **animal** box, a human has not
        already labelled it, and either it was never scored (no species model yet) or
        its winning score is below ``review_below``. That is precisely the set worth a
        human's attention: ``landscape``/``junk`` have no animal to name, ``multiple``
        has no single box that owns the image, and a human-labelled row is already
        settled.

        Ordered **least-confident first**, with never-scored rows ahead of scored ones,
        so the most informative labels come first. Each item carries the model's top-k
        guesses so the UI can offer them as one-click buttons — the fastest correct
        label is one the user only has to confirm.
        """
        limit = max(1, min(limit, 200))
        threshold = config.review_below
        where = (
            "i.status = 'done' "
            "AND COALESCE(i.label_source, 'model') <> 'human' "
            "AND i.label NOT IN ('landscape', 'junk', 'multiple') "
            "AND EXISTS (SELECT 1 FROM boxes b WHERE b.sha256 = i.sha256 AND b.cls = 'animal') "
            "AND (i.confidence IS NULL OR i.confidence < ?)"
        )
        with _ro() as conn:
            total = conn.execute(
                f"SELECT COUNT(*) AS n FROM images i WHERE {where}", (threshold,)
            ).fetchone()["n"]
            rows = conn.execute(
                f"SELECT i.* FROM images i WHERE {where} "
                # NULL (never scored) first, then ascending score: most informative first.
                "ORDER BY (i.confidence IS NOT NULL), i.confidence ASC, i.sha256 "
                "LIMIT ? OFFSET ?",
                (threshold, limit, offset),
            ).fetchall()
            items = [_image_payload(conn, row) for row in rows]
        return {
            "items": items,
            "total": total,
            "review_below": threshold,
            "known_labels": sorted(known_labels - set(RESERVED_LABELS)),
        }

    @app.get("/api/run")
    def api_run() -> dict[str, Any]:
        with _ro() as conn:
            row = conn.execute(
                "SELECT * FROM runs ORDER BY started_at DESC, run_id DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return {"run": None}
        return {"run": {k: row[k] for k in row.keys()}}

    # ---- helpers closed over config -------------------------------------- #

    def _readable_dest(sha256: str) -> Path:
        _validate_sha(sha256)
        with _ro() as conn:
            row = conn.execute(
                "SELECT status, dest_path FROM images WHERE sha256 = ?", (sha256,)
            ).fetchone()
        if row is None:
            raise HTTPException(404, "unknown image")
        reason = _no_file_reason(row)
        if reason is not None:
            raise HTTPException(
                409, {"error": "no readable materialized file for this image", "reason": reason}
            )
        return Path(row["dest_path"])

    return app


# --------------------------------------------------------------------------- #
# Module-level helpers (no app state)
# --------------------------------------------------------------------------- #


def _validate_sha(sha256: str) -> None:
    from fastapi import HTTPException

    if not _SHA256_RE.match(sha256):
        raise HTTPException(422, "sha256 must be 64 hex characters")


def _validate_label(label: str, known: set[str], allow_new: bool) -> None:
    from fastapi import HTTPException

    if not LABEL_RE.fullmatch(label) or label in RESERVED_LABELS:
        raise HTTPException(422, f"label {label!r} is not a legal, non-reserved label")
    if not allow_new and label not in known:
        raise HTTPException(
            422, f"unknown label {label!r}; start the GUI with --allow-new-labels to add it"
        )


def _no_file_reason(row: sqlite3.Row) -> str | None:
    if row["status"] == str(Status.PLANNED):
        return "planned"
    dest = row["dest_path"]
    if not dest:
        return "missing"
    path = Path(dest)
    if path.is_symlink() and not path.exists():
        return "dangling_symlink"
    if not path.exists():
        return "missing"
    return None


def _count_files_on_disk(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(
        1
        for entry in directory.iterdir()
        if not entry.name.startswith(".") and (entry.is_file() or entry.is_symlink())
    )


def _count_unscored(conn: sqlite3.Connection, clauses: list[str], params: list[Any]) -> int:
    where = " AND ".join([*clauses, "confidence IS NULL"])
    return conn.execute(f"SELECT COUNT(*) AS n FROM images WHERE {where}", params).fetchone()["n"]


def _image_payload(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    boxes = conn.execute(
        "SELECT * FROM boxes WHERE sha256 = ? ORDER BY idx", (row["sha256"],)
    ).fetchall()
    box_items = []
    for box in boxes:
        candidates = conn.execute(
            "SELECT rank, common, scientific, score FROM candidates WHERE box_id = ? ORDER BY rank",
            (box["id"],),
        ).fetchall()
        box_items.append(
            {
                "cls": box["cls"],
                "conf": box["conf"],
                "x0": box["x0"], "y0": box["y0"], "x1": box["x1"], "y1": box["y1"],
                "area_frac": box["area_frac"],
                "is_dominant": bool(box["is_dominant"]),
                "species_common": box["species_common"],
                "species_rank": box["species_rank"],
                "species_status": box["species_status"],
                "candidates": [dict(c) for c in candidates],
            }
        )
    return {
        "sha256": row["sha256"],
        "label": row["label"],
        "confidence": row["confidence"],
        "species_common": row["species_common"],
        "species_scientific": row["species_scientific"],
        "species_rank": row["species_rank"],
        "width": row["width"], "height": row["height"],
        "blur_score": row["blur_score"], "blur_ref_edge": row["blur_ref_edge"],
        "label_source": row["label_source"],
        "provider_status": row["provider_status"],
        "boxes": box_items,
    }


def _thumbnail(sha256: str, source: Path, output_root: Path) -> Path:
    """Generate (and cache) a 320 px JPEG thumbnail from ``dest_path`` (§6)."""
    from PIL import Image, ImageOps

    cache_dir = output_root / ".thumbs" / sha256[:2]
    cache_dir.mkdir(parents=True, exist_ok=True)
    thumb = cache_dir / f"{sha256}.jpg"
    if thumb.is_file():
        return thumb
    with Image.open(source) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((_THUMB_EDGE, _THUMB_EDGE))
        img.save(thumb, "JPEG", quality=85)
    return thumb


def _do_retag(config: Config, sha256: str, row: sqlite3.Row, new_label: str, note: Any) -> Path:
    """Record the override (I4) then perform the one rename (§5.8)."""
    from ..config import Mode

    old_dest = Path(row["dest_path"])
    mode = Mode(row["mode"]) if row["mode"] else config.mode
    # retry_writes=False: §5.9 wants contention to surface as a fast 409, not as a
    # browser request sitting through classify's 0.5/1/2/4/8 s backoff. The override
    # insert is the first write, so losing the lock changes nothing on disk.
    with Catalog.open(
        config.catalog_path,
        busy_timeout_ms=GUI_BUSY_TIMEOUT_MS,
        retry_writes=False,
    ) as catalog:
        catalog.insert_override(sha256, old_label=row["label"], new_label=new_label, note=note)
        catalog.update_image(
            sha256, label=new_label, label_source=str(LabelSource.HUMAN),
            status=str(Status.MATERIALIZING), dest_path=str(old_dest),
        )
        new_path = retag(
            old_dest, new_label=new_label, sha256=sha256,
            output_root=config.output_root, mode=mode,
        )
        catalog.update_image(sha256, status=str(Status.DONE), dest_path=str(new_path))
    return new_path


def _known_label_space(config: Config) -> set[str]:
    """Labels the GUI accepts without ``--allow-new-labels``: taxonomy + catalog."""
    labels = set(merged().keys()) | set(RESERVED_LABELS)
    try:
        with sqlite3.connect(f"file:{config.catalog_path}?mode=ro", uri=True) as conn:
            labels |= {
                r[0] for r in conn.execute("SELECT DISTINCT label FROM images WHERE label IS NOT NULL")
            }
    except sqlite3.Error:
        pass
    return labels
