"""The review GUI: a local FastAPI app over a classified output tree (§6).

Imported lazily by ``cli gui`` so the FastAPI/uvicorn import cost is not paid by
``classify``. ``create_app(config)`` builds the app for both ``uvicorn`` and the
e2e suite's in-process ``TestClient``.
"""

from __future__ import annotations
