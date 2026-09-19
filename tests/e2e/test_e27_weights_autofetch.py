"""E27: the detector checkpoint is fetched on first use, and verified before trust.

Added after a user ran the documented command on a fresh clone and got exit 3: the
default detector needs a 280 MB checkpoint that is not in git, and the tool refused
with a ``curl`` command instead of doing the fetch itself. "Download this manually
first" is not a working tool.

**Why auto-download does not violate I7.** I7 forbids silently substituting a
*different* value for a missing required one. Nothing is substituted here: the fetch
acquires the one artifact the code was written against, identified by the pinned byte
length and sha256, and **anything else is deleted rather than used**. The bytes land
at ``<weights>.part`` and are verified before being renamed, so an interrupted or
tampered download can never leave a file at the real path that a later run trusts.

``--no-download`` restores the fail-loud behaviour for offline or metered machines.
"""

from __future__ import annotations

import http.server
import tempfile
import threading
from pathlib import Path

import pytest

from animal_classifier.detect import megadetector as md
from animal_classifier.errors import AssetError


def _serve(body: bytes):
    """A local HTTP server returning ``body``, so no network is touched."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/md.pt"


def test_a_mismatched_download_is_rejected_and_cleaned_up(monkeypatch):
    """The safety property that makes auto-download acceptable at all."""
    server, url = _serve(b"definitely not a 280 MB yolov5 checkpoint")
    monkeypatch.setattr(md, "DOWNLOAD_URL", url)
    destination = Path(tempfile.mkdtemp()) / "md_v5a.0.0.pt"
    try:
        with pytest.raises(AssetError) as caught:
            md.ensure_weights(destination, allow_download=True)
    finally:
        server.shutdown()

    message = str(caught.value)
    assert md.EXPECTED_SHA256[:8] in message, "the message names the pinned hash"
    assert caught.value.exit_code == 3
    assert not destination.exists(), "a bad download must never take the real name"
    assert not destination.with_name(destination.name + ".part").exists(), (
        "the partial file is cleaned up"
    )


def test_no_download_fails_loudly_with_the_manual_command():
    """``--no-download``'s contract: refuse, and say exactly how to fix it."""
    destination = Path(tempfile.mkdtemp()) / "md_v5a.0.0.pt"
    with pytest.raises(AssetError) as caught:
        md.ensure_weights(destination, allow_download=False)

    message = str(caught.value)
    assert caught.value.exit_code == 3
    assert "--no-download" in message
    assert "curl" in message and md.DOWNLOAD_URL in message
    assert not destination.exists()


def test_an_already_present_checkpoint_is_verified_not_refetched(monkeypatch):
    """A second run must not re-download; it re-verifies what is on disk."""
    calls: list[str] = []

    def _fail_if_called(url, destination):
        calls.append(url)
        raise AssertionError("must not download when the file is already present")

    monkeypatch.setattr(md, "_download", _fail_if_called)

    real = Path("models/md_v5a.0.0.pt")
    if not (real.is_file() and real.stat().st_size == md.EXPECTED_BYTES):
        pytest.skip("the real checkpoint is not present to verify against")

    md.ensure_weights(real, allow_download=True)  # must not raise, must not fetch
    assert calls == []


def test_a_corrupt_existing_checkpoint_is_rejected(monkeypatch):
    """An on-disk file of the wrong size is refused rather than unpickled (§10.1)."""
    monkeypatch.setattr(
        md, "_download", lambda url, dest: pytest.fail("should not download")
    )
    bogus = Path(tempfile.mkdtemp()) / "md_v5a.0.0.pt"
    bogus.write_bytes(b"truncated")
    with pytest.raises(AssetError, match="bytes, expected"):
        md.ensure_weights(bogus, allow_download=True)


def test_classify_help_documents_the_opt_out(run_cli):
    result = run_cli(["classify", "--help"])
    assert result.returncode == 0
    assert "--no-download" in result.stdout
