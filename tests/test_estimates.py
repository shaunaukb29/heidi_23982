import numpy as np

from carithm_inspector.analysis import locate_vehicle_part, priority_from, severity_from_detection
from carithm_inspector.domain import DamageType, Detection, Priority, Severity
from carithm_inspector.estimates import estimate
from carithm_inspector.visualization import render_inspection_overlay


def test_broken_lamp_is_not_marked_driveable() -> None:
    result = estimate(Detection(DamageType.LAMP_BROKEN, 0.84, (0, 0, 100, 100), 1000, 800))
    assert result.driveable is False
    assert result.low_usd < result.high_usd
    assert result.priority in {Priority.MODERATE, Priority.SAFETY_CRITICAL}


def test_larger_scratch_has_a_higher_estimate() -> None:
    small_detection = Detection(DamageType.SCRATCH, 0.9, (0, 0, 20, 20), 1000, 800)
    large_detection = Detection(DamageType.SCRATCH, 0.9, (0, 0, 300, 300), 1000, 800)
    small = estimate(small_detection)
    large = estimate(large_detection)
    assert large.low_usd > small.low_usd
    assert large_detection.area_ratio > small_detection.area_ratio


def test_mask_area_used_for_severity() -> None:
    mask = np.zeros((800, 1000), dtype=bool)
    mask[390:410, 490:510] = True
    detection = Detection(DamageType.SCRATCH, 0.92, (490, 390, 510, 410), 1000, 800, mask=mask)
    result = estimate(detection)
    assert result.severity == Severity.MINOR
    assert detection.area_ratio == 0.0005


def test_tire_flat_is_safety_critical() -> None:
    detection = Detection(DamageType.TIRE_FLAT, 0.95, (100, 600, 250, 750), 1000, 800)
    result = estimate(detection)
    assert result.priority == Priority.SAFETY_CRITICAL
    assert "Wheel" in result.vehicle_part.location


def test_cost_breakdown_sums_to_midpoint() -> None:
    detection = Detection(DamageType.DENT, 0.88, (100, 200, 400, 500), 1000, 800)
    result = estimate(detection)
    breakdown = result.cost_breakdown
    assert breakdown.total_usd == breakdown.labour_usd + breakdown.paint_usd + breakdown.parts_usd
    assert breakdown.labour_usd >= 0
    assert result.repair_steps


def test_locate_front_bumper_from_lower_center_bbox() -> None:
    detection = Detection(DamageType.SCRATCH, 0.9, (400, 700, 600, 780), 1000, 800)
    part = locate_vehicle_part(detection)
    assert "Bumper" in part.location


def test_overlay_renders_without_mask() -> None:
    from PIL import Image

    image = Image.new("RGB", (640, 480), color=(120, 120, 120))
    detection = Detection(DamageType.SCRATCH, 0.91, (100, 100, 220, 180), 640, 480)
    overlay = render_inspection_overlay(image, [detection], selected_index=0)
    assert overlay.size == image.size


def test_priority_from_cosmetic_scratch() -> None:
    detection = Detection(DamageType.SCRATCH, 0.9, (0, 0, 10, 10), 1000, 800)
    severity = severity_from_detection(detection)
    assert priority_from(detection, severity, driveable=True) == Priority.COSMETIC
