"""
Carithm AI — Semantic Part Mapping
====================================

Replaces the coordinate/bbox-based `locate_vehicle_part` heuristic with a
pixel-level, segmentation-driven resolution: the damage mask is intersected
against a per-pixel part-segmentation map, and the best-overlapping part
wins (argmax of intersection area).

Design notes
------------
Most of the special-casing in the legacy heuristic (tire -> wheel,
lamp -> headlight vs. taillight) existed only because a bounding-box
center has no notion of *what a region actually is* — it had to be
inferred from (damage_type, normalized_x, normalized_y, view_angle).
A real part-segmentation model removes that problem structurally: it
should emit HEADLIGHT_LEFT / TAILLIGHT_RIGHT / WHEEL_FRONT_LEFT etc. as
distinct classes, so damage type no longer needs to disambiguate part
identity — the mask overlap does it directly.

What is *not* discarded: the legacy bbox heuristic is kept, verbatim in
spirit, as an explicit fallback for:
    - detections with `mask is None` (older / box-only checkpoints)
    - a part-segmentation map that isn't available for this frame
    - semantic resolution that fails to clear `min_overlap_ratio`
      (e.g. damage sits on background / segmentation gap)

This means nothing silently regresses during the migration — every
resolution records *how* it was resolved via `ResolutionMethod`, which
should be logged/monitored so you can watch the fallback rate trend
toward zero as segmentation coverage improves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import Mapping, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .domain import Detection, DamageType, VehiclePart, ViewAngle

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Part taxonomy
# --------------------------------------------------------------------------- #


class PartID(IntEnum):
    """Canonical part classes. Values double as the PartSegmenter's
    per-pixel class indices, so this enum is the label schema contract
    between the segmentation model and everything downstream.
    """

    BACKGROUND = 0
    FRONT_BUMPER = auto()
    REAR_BUMPER = auto()
    HOOD = auto()
    TRUNK = auto()
    ROOF = auto()
    LEFT_DOOR = auto()
    RIGHT_DOOR = auto()
    FRONT_FENDER = auto()
    REAR_FENDER = auto()
    WINDSHIELD = auto()
    REAR_WINDSHIELD = auto()
    HEADLIGHT_LEFT = auto()
    HEADLIGHT_RIGHT = auto()
    TAILLIGHT_LEFT = auto()
    TAILLIGHT_RIGHT = auto()
    WHEEL_FRONT_LEFT = auto()
    WHEEL_FRONT_RIGHT = auto()
    WHEEL_REAR_LEFT = auto()
    WHEEL_REAR_RIGHT = auto()
    BODY_STRUCTURE = auto()
    UNKNOWN = auto()


# Single source of truth for part metadata — replaces the old pattern of
# constructing `VehiclePart(...)` inline, ad hoc, at every heuristic branch.
PART_CATALOG: Mapping[PartID, VehiclePart] = {
    PartID.FRONT_BUMPER: VehiclePart("Front Bumper", "Plastic bumper cover", "Plastic"),
    PartID.REAR_BUMPER: VehiclePart("Rear Bumper", "Plastic bumper cover", "Plastic"),
    PartID.HOOD: VehiclePart("Hood", "Steel panel", "Steel"),
    PartID.TRUNK: VehiclePart("Trunk", "Steel panel", "Steel"),
    PartID.ROOF: VehiclePart("Roof", "Roof panel", "Steel"),
    PartID.LEFT_DOOR: VehiclePart("Left Door", "Steel door panel", "Steel"),
    PartID.RIGHT_DOOR: VehiclePart("Right Door", "Steel door panel", "Steel"),
    PartID.FRONT_FENDER: VehiclePart("Front Fender/Quarter Panel", "Steel body panel", "Steel"),
    PartID.REAR_FENDER: VehiclePart("Rear Fender/Quarter Panel", "Steel body panel", "Steel"),
    PartID.WINDSHIELD: VehiclePart("Windshield", "Laminated glass", "Glass"),
    PartID.REAR_WINDSHIELD: VehiclePart("Rear Windshield", "Tempered glass", "Glass"),
    PartID.HEADLIGHT_LEFT: VehiclePart("Left Headlight", "Lamp assembly", "Polycarbonate"),
    PartID.HEADLIGHT_RIGHT: VehiclePart("Right Headlight", "Lamp assembly", "Polycarbonate"),
    PartID.TAILLIGHT_LEFT: VehiclePart("Left Tail Light", "Lamp assembly", "Polycarbonate"),
    PartID.TAILLIGHT_RIGHT: VehiclePart("Right Tail Light", "Lamp assembly", "Polycarbonate"),
    PartID.WHEEL_FRONT_LEFT: VehiclePart("Front Left Wheel", "Tire and wheel assembly", "Rubber / Alloy"),
    PartID.WHEEL_FRONT_RIGHT: VehiclePart("Front Right Wheel", "Tire and wheel assembly", "Rubber / Alloy"),
    PartID.WHEEL_REAR_LEFT: VehiclePart("Rear Left Wheel", "Tire and wheel assembly", "Rubber / Alloy"),
    PartID.WHEEL_REAR_RIGHT: VehiclePart("Rear Right Wheel", "Tire and wheel assembly", "Rubber / Alloy"),
    PartID.BODY_STRUCTURE: VehiclePart("Body Structure", "Chassis component", "Steel"),
    PartID.UNKNOWN: VehiclePart("Unresolved", "Unresolved component", "Unknown"),
}


class ResolutionMethod(str, Enum):
    """Records how a detection's part was determined — track/monitor this
    in production so the fallback rate can be watched as segmentation
    coverage improves."""

    SEMANTIC_OVERLAP = "semantic_overlap"
    LEGACY_HEURISTIC = "legacy_heuristic"


@dataclass(frozen=True, slots=True)
class ResolvedDetection:
    """A `Detection` fused with its resolved vehicle part."""

    detection: Detection
    part_id: PartID
    vehicle_part: VehiclePart
    overlap_ratio: float
    method: ResolutionMethod


# --------------------------------------------------------------------------- #
# Semantic (mask-overlap) resolution
# --------------------------------------------------------------------------- #


def best_fit_part(
    damage_mask: NDArray[np.bool_],
    part_mask: NDArray[np.int_],
    *,
    min_overlap_ratio: float = 0.05,
) -> tuple[PartID, float]:
    """Resolve the best-fitting `PartID` for a damage mask via pixel overlap.

    For every part class present under the damage mask, count how many
    damage pixels fall on it, then take the argmax. No coordinates, no
    view-angle branching — geometry is entirely delegated to the
    segmentation model.

    Returns:
        (PartID.UNKNOWN, 0.0) if the masks don't overlap, or the best
        overlap doesn't clear `min_overlap_ratio` (e.g. damage sits
        entirely on background/segmentation gap).
    """
    if damage_mask.shape != part_mask.shape:
        raise ValueError(
            f"Shape mismatch: damage_mask {damage_mask.shape} vs part_mask {part_mask.shape}"
        )

    total_damage_px = int(np.count_nonzero(damage_mask))
    if total_damage_px == 0:
        return PartID.UNKNOWN, 0.0

    overlapping = part_mask[damage_mask]
    part_ids, counts = np.unique(overlapping, return_counts=True)

    non_bg = part_ids != PartID.BACKGROUND.value
    part_ids, counts = part_ids[non_bg], counts[non_bg]
    if part_ids.size == 0:
        return PartID.UNKNOWN, 0.0

    best_idx = int(np.argmax(counts))
    overlap_ratio = int(counts[best_idx]) / total_damage_px

    if overlap_ratio < min_overlap_ratio:
        return PartID.UNKNOWN, overlap_ratio

    try:
        best_part = PartID(int(part_ids[best_idx]))
    except ValueError:
        best_part = PartID.UNKNOWN

    return best_part, overlap_ratio


# --------------------------------------------------------------------------- #
# Legacy fallback (kept verbatim in spirit — bbox + view-angle heuristic)
# --------------------------------------------------------------------------- #


def _horizontal_zone(cx: float) -> str:
    if cx < 0.20:
        return "Far Left"
    if cx < 0.40:
        return "Left"
    if cx < 0.60:
        return "Center"
    if cx < 0.80:
        return "Right"
    return "Far Right"


def locate_vehicle_part_legacy(
    detection: Detection, view_angle: ViewAngle = ViewAngle.UNKNOWN
) -> VehiclePart:
    """The original bbox/coordinate heuristic. Kept as an explicit fallback
    for when no usable mask/part-segmentation is available — not for new
    code paths to depend on directly.
    """
    x1, y1, x2, y2 = detection.bbox
    cx = ((x1 + x2) / 2) / detection.image_width
    cy = ((y1 + y2) / 2) / detection.image_height
    damage = detection.damage_type

    zone = _horizontal_zone(cx)
    if view_angle == ViewAngle.RIGHT_SIDE:
        is_front_zone = "Right" in zone
        is_rear_zone = "Left" in zone
    elif view_angle == ViewAngle.LEFT_SIDE:
        is_front_zone = "Left" in zone
        is_rear_zone = "Right" in zone
    else:
        is_front_zone = "Left" in zone
        is_rear_zone = "Right" in zone

    if damage == DamageType.TIRE_FLAT:
        side = "Front" if is_front_zone else "Rear"
        return VehiclePart(f"{side} Wheel", "Tire and wheel assembly", "Rubber / Alloy")

    if damage == DamageType.LAMP_BROKEN:
        lamp_type = "Headlight" if (view_angle == ViewAngle.FRONT or is_front_zone) else "Tail Light"
        return VehiclePart(lamp_type, "Lamp assembly", "Polycarbonate")

    if cy > 0.40 and (is_front_zone or is_rear_zone):
        if damage in (DamageType.SCRATCH, DamageType.DENT):
            prefix = "Front" if is_front_zone else "Rear"
            if cy < 0.65:
                return VehiclePart(f"{prefix} Fender/Quarter Panel", "Steel body panel", "Steel")
            return VehiclePart(f"{prefix} Bumper", "Plastic bumper cover", "Plastic")

    if cy < 0.12:
        return VehiclePart("Roof", "Roof panel", "Steel")
    if cy < 0.28:
        return VehiclePart("Hood / Trunk", "Steel panel", "Steel")
    if cy < 0.70:
        side = "Left" if view_angle == ViewAngle.LEFT_SIDE else "Right"
        return VehiclePart(f"{side} Door", "Steel door panel", "Steel")

    return VehiclePart("Body Structure", "Chassis component", "Steel")


# --------------------------------------------------------------------------- #
# Unified entry point
# --------------------------------------------------------------------------- #


def resolve_vehicle_part(
    detection: Detection,
    part_mask: NDArray[np.int_] | None,
    view_angle: ViewAngle = ViewAngle.UNKNOWN,
    *,
    min_overlap_ratio: float = 0.05,
) -> ResolvedDetection:
    """Resolve `detection` to a `VehiclePart`, preferring semantic mask
    overlap and falling back to the legacy bbox heuristic when semantic
    resolution isn't possible or doesn't clear the confidence bar.
    """
    if detection.mask is not None and part_mask is not None:
        part_id, overlap_ratio = best_fit_part(
            detection.mask, part_mask, min_overlap_ratio=min_overlap_ratio
        )
        if part_id is not PartID.UNKNOWN:
            return ResolvedDetection(
                detection=detection,
                part_id=part_id,
                vehicle_part=PART_CATALOG[part_id],
                overlap_ratio=overlap_ratio,
                method=ResolutionMethod.SEMANTIC_OVERLAP,
            )
        logger.info(
            "Semantic overlap unresolved (ratio=%.3f) for %s — falling back to legacy heuristic.",
            overlap_ratio,
            detection.damage_type,
        )
    else:
        logger.debug(
            "No mask/part_mask available for %s — using legacy heuristic.",
            detection.damage_type,
        )

    legacy_part = locate_vehicle_part_legacy(detection, view_angle)
    return ResolvedDetection(
        detection=detection,
        part_id=PartID.UNKNOWN,
        vehicle_part=legacy_part,
        overlap_ratio=0.0,
        method=ResolutionMethod.LEGACY_HEURISTIC,
    )


def resolve_all(
    detections: list[Detection],
    part_mask: NDArray[np.int_] | None,
    view_angle: ViewAngle = ViewAngle.UNKNOWN,
    *,
    min_overlap_ratio: float = 0.05,
) -> list[ResolvedDetection]:
    return [
        resolve_vehicle_part(d, part_mask, view_angle, min_overlap_ratio=min_overlap_ratio)
        for d in detections
    ]


# --------------------------------------------------------------------------- #
# Model interfaces (structural typing — no implementation here)
# --------------------------------------------------------------------------- #


class OrientationModel(Protocol):
    def predict(self, image: Image.Image) -> ViewAngle: ...


class PartSegmenter(Protocol):
    def predict(self, image: Image.Image) -> NDArray[np.int_]:
        """Return an H×W int array where each pixel holds a `PartID` value."""
        ...


class DamageDetectorLike(Protocol):
    def inspect(self, image: Image.Image) -> list[Detection]: ...


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class InspectionResult:
    view_angle: ViewAngle
    part_mask: NDArray[np.int_] | None
    resolved: list[ResolvedDetection]


class InspectionPipeline:
    """Runs orientation, part segmentation, and damage detection, then
    fuses them via `resolve_all`. Wired against the real `Detection` /
    `ViewAngle` domain types and the existing `inspect()`-style detector
    adapters — no protocol drift from the rest of the codebase.
    """

    def __init__(
        self,
        orientation_model: OrientationModel,
        part_segmenter: PartSegmenter,
        damage_detector: DamageDetectorLike,
        *,
        min_overlap_ratio: float = 0.05,
    ) -> None:
        self._orientation_model = orientation_model
        self._part_segmenter = part_segmenter
        self._damage_detector = damage_detector
        self._min_overlap_ratio = min_overlap_ratio

    def run(self, image: Image.Image) -> InspectionResult:
        view_angle = self._safe_predict_orientation(image)
        part_mask = self._safe_predict_part_mask(image)
        detections = self._damage_detector.inspect(image)

        resolved = resolve_all(
            detections, part_mask, view_angle, min_overlap_ratio=self._min_overlap_ratio
        )
        return InspectionResult(view_angle=view_angle, part_mask=part_mask, resolved=resolved)

    def _safe_predict_orientation(self, image: Image.Image) -> ViewAngle:
        try:
            return self._orientation_model.predict(image)
        except Exception:
            logger.exception("Orientation head failed; defaulting to UNKNOWN.")
            return ViewAngle.UNKNOWN

    def _safe_predict_part_mask(self, image: Image.Image) -> NDArray[np.int_] | None:
        try:
            return self._part_segmenter.predict(image)
        except Exception:
            logger.exception("Part segmentation head failed; falling back to legacy heuristic for this frame.")
            return None
