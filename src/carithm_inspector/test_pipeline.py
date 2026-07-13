"""
Smoke tests for the semantic part-mapping and estimation refactor.
Run with: pytest test_pipeline.py -v

These use fakes/synthetic masks — no model weights required.
"""

from __future__ import annotations

import numpy as np
import pytest

from carithm.domain import DamageType, Detection, ViewAngle
from carithm.part_mapping import (
    PartID,
    ResolutionMethod,
    best_fit_part,
    resolve_vehicle_part,
)
from carithm.estimation import is_driveable, estimate_for


def make_detection(damage_type, mask=None, bbox=(100, 100, 200, 200), w=640, h=480):
    return Detection(
        damage_type=damage_type,
        confidence=0.9,
        bbox=bbox,
        image_width=w,
        image_height=h,
        mask=mask,
    )


class TestBestFitPart:
    def test_clean_overlap_picks_correct_part(self):
        part_mask = np.zeros((100, 100), dtype=np.int_)
        part_mask[:, :50] = PartID.LEFT_DOOR.value
        part_mask[:, 50:] = PartID.FRONT_FENDER.value

        damage_mask = np.zeros((100, 100), dtype=np.bool_)
        damage_mask[40:60, 10:40] = True  # entirely inside LEFT_DOOR region

        part_id, ratio = best_fit_part(damage_mask, part_mask)
        assert part_id == PartID.LEFT_DOOR
        assert ratio == pytest.approx(1.0)

    def test_split_damage_picks_majority_part(self):
        part_mask = np.zeros((100, 100), dtype=np.int_)
        part_mask[:, :50] = PartID.LEFT_DOOR.value
        part_mask[:, 50:] = PartID.FRONT_FENDER.value

        damage_mask = np.zeros((100, 100), dtype=np.bool_)
        damage_mask[40:60, 30:70] = True  # 20px on door, 20px on fender -> tie broken by argmax order

        part_id, ratio = best_fit_part(damage_mask, part_mask)
        assert part_id in (PartID.LEFT_DOOR, PartID.FRONT_FENDER)
        assert 0.0 < ratio <= 1.0

    def test_damage_entirely_on_background_returns_unknown(self):
        part_mask = np.zeros((100, 100), dtype=np.int_)  # all BACKGROUND
        damage_mask = np.zeros((100, 100), dtype=np.bool_)
        damage_mask[10:20, 10:20] = True

        part_id, ratio = best_fit_part(damage_mask, part_mask)
        assert part_id == PartID.UNKNOWN
        assert ratio == 0.0

    def test_empty_damage_mask_returns_unknown(self):
        part_mask = np.full((50, 50), PartID.HOOD.value, dtype=np.int_)
        damage_mask = np.zeros((50, 50), dtype=np.bool_)

        part_id, ratio = best_fit_part(damage_mask, part_mask)
        assert part_id == PartID.UNKNOWN
        assert ratio == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            best_fit_part(np.zeros((10, 10), dtype=np.bool_), np.zeros((20, 20), dtype=np.int_))

    def test_below_min_overlap_ratio_falls_to_unknown(self):
        part_mask = np.zeros((100, 100), dtype=np.int_)
        part_mask[0:1, 0:1] = PartID.HOOD.value  # tiny sliver of real part

        damage_mask = np.zeros((100, 100), dtype=np.bool_)
        damage_mask[0:50, 0:50] = True  # mostly background, 1px overlaps HOOD

        part_id, ratio = best_fit_part(damage_mask, part_mask, min_overlap_ratio=0.5)
        assert part_id == PartID.UNKNOWN


class TestResolveVehiclePart:
    def test_uses_semantic_path_when_masks_available(self):
        part_mask = np.full((480, 640), PartID.HOOD.value, dtype=np.int_)
        damage_mask = np.zeros((480, 640), dtype=np.bool_)
        damage_mask[100:150, 100:150] = True

        detection = make_detection(DamageType.DENT, mask=damage_mask)
        resolved = resolve_vehicle_part(detection, part_mask)

        assert resolved.method == ResolutionMethod.SEMANTIC_OVERLAP
        assert resolved.part_id == PartID.HOOD

    def test_falls_back_when_mask_is_none(self):
        detection = make_detection(DamageType.SCRATCH, mask=None)
        resolved = resolve_vehicle_part(detection, part_mask=None, view_angle=ViewAngle.LEFT_SIDE)

        assert resolved.method == ResolutionMethod.LEGACY_HEURISTIC
        assert resolved.vehicle_part is not None

    def test_falls_back_when_part_mask_is_none(self):
        damage_mask = np.zeros((480, 640), dtype=np.bool_)
        damage_mask[10:20, 10:20] = True
        detection = make_detection(DamageType.DENT, mask=damage_mask)

        resolved = resolve_vehicle_part(detection, part_mask=None)
        assert resolved.method == ResolutionMethod.LEGACY_HEURISTIC

    def test_falls_back_when_semantic_overlap_unresolved(self):
        part_mask = np.zeros((480, 640), dtype=np.int_)  # all background
        damage_mask = np.zeros((480, 640), dtype=np.bool_)
        damage_mask[10:20, 10:20] = True
        detection = make_detection(DamageType.DENT, mask=damage_mask)

        resolved = resolve_vehicle_part(detection, part_mask)
        assert resolved.method == ResolutionMethod.LEGACY_HEURISTIC


class TestDriveability:
    def test_flat_tire_not_driveable(self):
        detection = make_detection(DamageType.TIRE_FLAT)
        resolved = resolve_vehicle_part(detection, part_mask=None)
        assert is_driveable(resolved) is False

    def test_scratch_is_driveable(self):
        detection = make_detection(DamageType.SCRATCH)
        resolved = resolve_vehicle_part(detection, part_mask=None, view_angle=ViewAngle.FRONT)
        assert is_driveable(resolved) is True


class TestEstimateFor:
    def test_produces_consistent_cost_bounds(self):
        detection = make_detection(DamageType.DENT)
        resolved = resolve_vehicle_part(detection, part_mask=None, view_angle=ViewAngle.FRONT)
        estimate = estimate_for(resolved)

        assert estimate.low_cost <= estimate.cost_breakdown.total_cost <= estimate.high_cost
        assert estimate.currency == "USD"
        assert estimate.severity is not None
        assert estimate.priority is not None

    def test_tire_flat_estimate_is_safety_critical(self):
        detection = make_detection(DamageType.TIRE_FLAT)
        resolved = resolve_vehicle_part(detection, part_mask=None)
        estimate = estimate_for(resolved)

        assert estimate.driveable is False
        assert estimate.priority.value == "safety_critical"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
