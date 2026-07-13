"""
Carithm AI — End-to-End Inspection Entrypoint
=================================================

Wires together:
    - damage_detector.load_detector()      (your existing YOLO/MMDetection adapters)
    - temporary_models.HeuristicOrientationModel / NullPartSegmenter
    - part_mapping.InspectionPipeline
    - estimation.estimates_for_all

This is the "ship it now" entrypoint. It runs correctly TODAY on the
legacy bbox heuristic (via NullPartSegmenter) and upgrades automatically,
with zero call-site changes, the moment a trained PartSegmenter and a
trained OrientationModel are swapped in below.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from .damage_detector import load_detector
from .domain import Estimate
from .estimation import estimates_for_all
from .part_mapping import InspectionPipeline
from .temporary_models import HeuristicOrientationModel, NullPartSegmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_pipeline() -> InspectionPipeline:
    """Construct the pipeline with whatever is available today.

    Swap `HeuristicOrientationModel()` -> a trained orientation classifier
    and `NullPartSegmenter()` -> a trained part segmenter as those become
    available. No other code in this file or in `part_mapping.py` needs
    to change when you do.
    """
    return InspectionPipeline(
        orientation_model=HeuristicOrientationModel(),
        part_segmenter=NullPartSegmenter(),
        damage_detector=load_detector(),
    )


def inspect_image(image_path: Path, pipeline: InspectionPipeline | None = None) -> list[Estimate]:
    """Run the full pipeline on a single image file and return estimates."""
    pipeline = pipeline or build_pipeline()
    image = Image.open(image_path)

    result = pipeline.run(image)
    estimates = estimates_for_all(result.resolved)

    logger.info(
        "Inspected %s: view_angle=%s, %d detection(s) -> %d estimate(s)",
        image_path.name,
        result.view_angle.value,
        len(result.resolved),
        len(estimates),
    )
    return estimates


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m carithm.inspect <image_path>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"No such file: {path}")
        sys.exit(1)

    results = inspect_image(path)
    if not results:
        print("No damage detected.")
    for est in results:
        print(
            f"[{est.priority.value.upper()}] {est.vehicle_part.location} — "
            f"{est.severity.value} ({est.complexity} complexity)\n"
            f"  Cost: ${est.low_cost:.2f}–${est.high_cost:.2f} {est.currency}\n"
            f"  Shop time: {est.shop_time_low_hours}–{est.shop_time_high_hours} hrs\n"
            f"  Driveable: {est.driveable} — {est.safety_note}\n"
            f"  Steps: {', '.join(est.repair_steps)}\n"
        )
