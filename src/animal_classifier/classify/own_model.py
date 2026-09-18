"""Our species classifier, run per animal crop (DESIGN.md §5.5).

Loads a ``.acmodel`` artifact (§7.1) and classifies each animal crop: batch the
crops of one image, forward, ``softmax(logits / temperature)``, take the top-5. The
temperature is read from the artifact **without a fallback** — a missing key was
already a fatal load error (§7.1), so here it is simply used.

The output is a :class:`~animal_classifier.decide.SpeciesPrediction` for the top-1
of each crop, which is exactly what :func:`animal_classifier.decide.decide` consumes.
A degenerate crop (no pixels) is never sent here — the pipeline leaves its
``species_status='degenerate'`` and it becomes ``unknown`` at decide time (§5.3).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from ..decide import SpeciesPrediction
from .artifact import Artifact, load

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One ranked prediction for a crop (for ``candidates`` rows / the GUI top-3)."""

    rank: int
    common: str
    scientific: str | None
    score: float


class SpeciesClassifier:
    """An ``.acmodel`` artifact wrapped as a per-crop classifier."""

    def __init__(self, artifact: Artifact) -> None:
        self._artifact = artifact

    @classmethod
    def from_path(cls, path: Path) -> SpeciesClassifier:
        return cls(load(path))

    @property
    def model_id(self) -> str:
        return self._artifact.model_id

    def classify(self, crops):
        """Top prediction + ranked candidates for each crop image.

        ``crops`` is a sequence of PIL images (the pipeline's degenerate crops are
        excluded upstream). Returns a list aligned with ``crops`` of
        ``(SpeciesPrediction, [Candidate, ...])`` — or ``(None, [])`` for an empty
        input, so the caller never special-cases the no-crops image.
        """
        import torch
        from torchvision import transforms

        if not crops:
            return []

        art = self._artifact
        transform = transforms.Compose(
            [
                transforms.Resize(round(art.input_size * 1.14)),
                transforms.CenterCrop(art.input_size),
                transforms.ToTensor(),
                transforms.Normalize(art.mean, art.std),
            ]
        )
        batch = torch.stack([transform(image.convert("RGB")) for image in crops])
        with torch.inference_mode():
            logits = art.module(batch) / art.temperature
            probs = torch.softmax(logits, dim=1)

        results = []
        maxk = min(5, probs.size(1))
        top_scores, top_indices = probs.topk(maxk, dim=1)
        for scores, indices in zip(top_scores.tolist(), top_indices.tolist(), strict=True):
            candidates = [
                Candidate(
                    rank=rank + 1,
                    common=art.labels[idx].common,
                    scientific=art.labels[idx].scientific,
                    score=score,
                )
                for rank, (idx, score) in enumerate(zip(indices, scores, strict=True))
            ]
            best = art.labels[indices[0]]
            prediction = SpeciesPrediction(
                slug=best.slug,
                common=best.common,
                scientific=best.scientific,
                rank=best.rank,
                conf=scores[0],
            )
            results.append((prediction, candidates))
        return results
