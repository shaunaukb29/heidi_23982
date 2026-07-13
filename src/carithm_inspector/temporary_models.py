"""
Carithm AI — Temporary Model Stand-ins
=========================================

`OrientationModel` and `PartSegmenter` are architecture contracts, not
trained models — I cannot produce real weights here. What's below lets
the full pipeline RUN TODAY without either:

    - `HeuristicOrientationModel`: a rule-based (not learned) guess at
      Front/Rear/Side from image aspect ratio. Crude, but unblocks
      shipping — swap for a real classifier when you have one trained.
    - `NullPartSegmenter`: always returns `None`, which makes
      `resolve_vehicle_part` take the legacy bbox-heuristic path for
      every detection. This is exactly the intended fallback behavior
      from `part_mapping.py` — the app runs correctly on the legacy
      heuristic alone, with zero code changes needed later when a real
      segmenter is dropped in (just swap the class passed into
      `InspectionPipeline`).

Do not mistake `HeuristicOrientationModel` for a trained model in any
metrics/monitoring you set up — track its usage the same way you'd
track the legacy-heuristic fallback rate.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .domain import ViewAngle

logger = logging.getLogger(__name__)


class HeuristicOrientationModel:
    """Rule-based ViewAngle guess from image aspect ratio.

    Real vehicle-orientation classification needs a trained model (this
    was head #1 in the target architecture) — this is a stopgap so the
    pipeline has *some* orientation signal instead of hardcoding UNKNOWN
    everywhere, which would silently disable orientation-dependent
    branches of the legacy heuristic (front/rear door siding, etc).

    Heuristic: side-profile shots of a car are typically wider than
    tall (car length >> car height in frame); front/rear shots taken
    head-on tend toward a more square or portrait-ish aspect ratio.
    This is NOT reliable — replace as soon as a trained classifier
    exists.
    """

    SIDE_ASPECT_THRESHOLD: float = 1.35

    def predict(self, image: Image.Image) -> ViewAngle:
        width, height = image.size
        if height == 0:
            return ViewAngle.UNKNOWN

        aspect_ratio = width / height
        if aspect_ratio >= self.SIDE_ASPECT_THRESHOLD:
            # Can't distinguish left vs. right from aspect ratio alone.
            logger.debug("HeuristicOrientationModel: wide aspect (%.2f) -> LEFT_SIDE (unverified)", aspect_ratio)
            return ViewAngle.LEFT_SIDE

        logger.debug("HeuristicOrientationModel: aspect %.2f inconclusive -> UNKNOWN", aspect_ratio)
        return ViewAngle.UNKNOWN


class NullPartSegmenter:
    """Always returns `None` for the part mask.

    Wiring this into `InspectionPipeline` means every detection takes
    the legacy bbox-heuristic fallback path in `resolve_vehicle_part` —
    i.e. the app behaves exactly like the pre-refactor pipeline until a
    real `PartSegmenter` (head #2) is trained and swapped in.
    """

    def predict(self, image: Image.Image) -> NDArray[np.int_] | None:
        return None
