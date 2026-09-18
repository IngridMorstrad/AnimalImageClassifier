"""Species classification: the ``.acmodel`` artifact and the own-model head (§5.5, §7.1).

Import from this package. The heavy torch-backed pieces (:mod:`artifact`,
:mod:`own_model`) are imported lazily by their callers so that
``classify --detector scripted`` with no species model, and the GUI, never pay the
torch import cost.
"""

from __future__ import annotations
