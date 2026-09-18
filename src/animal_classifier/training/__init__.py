"""The training subsystem: manifest, dataset, trainer, eval (DESIGN.md §7).

Imported lazily by ``cli`` only when ``train``/``eval`` run, because it pulls in
torch and torchvision — a cost ``classify --detector scripted`` and the GUI must
not pay. Import the submodules directly rather than eagerly re-exporting them here.
"""

from __future__ import annotations
