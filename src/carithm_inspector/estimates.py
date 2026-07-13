from __future__ import annotations

from .analysis import (
    locate_vehicle_part,
    priority_from,
    repair_steps_for,
    severity_from_detection,
    shop_time_for,
)
from .domain import CostBreakdown, DamageType, Detection, Estimate, Severity, ViewAngle


# Baselines are consumer repair ranges, before visible-area adjustment.
_REPAIR_GUIDANCE: dict[DamageType, tuple[int, int, str, bool, str, tuple[str, ...]]] = {
    DamageType.SCRATCH: (180, 350, "low", True, "Cosmetic damage can hide sharp edges.", ("paint", "clear coat")),
    DamageType.DENT: (200, 650, "medium", True, "Check that panels and doors still operate normally.", ("body panel", "paint")),
    DamageType.CRACK: (250, 900, "medium", False, "A crack near safety glass, lights, or a bumper mount needs inspection before driving.", ("body panel", "bumper cover")),
    DamageType.GLASS_SHATTER: (300, 1_200, "high", False, "Broken vehicle glass can obstruct vision and leave loose shards.", ("glass assembly", "weather seal")),
    DamageType.LAMP_BROKEN: (180, 1_100, "medium", False, "Do not drive at night or in poor visibility with a damaged lamp.", ("lamp assembly", "bulb or LED module")),
    DamageType.TIRE_FLAT: (120, 450, "medium", False, "Do not drive on a flat tire. Fit the spare or arrange recovery.", ("tire", "valve stem", "wheel repair")),
}

# Share of total repair cost by category (labour, paint, parts).
_COST_SPLIT: dict[DamageType, tuple[float, float, float]] = {
    DamageType.SCRATCH: (0.35, 0.55, 0.10),
    DamageType.DENT: (0.50, 0.40, 0.10),
    DamageType.CRACK: (0.45, 0.30, 0.25),
    DamageType.GLASS_SHATTER: (0.30, 0.05, 0.65),
    DamageType.LAMP_BROKEN: (0.25, 0.05, 0.70),
    DamageType.TIRE_FLAT: (0.20, 0.00, 0.80),
}


def _cost_breakdown(damage_type: DamageType, total: int) -> CostBreakdown:
    labour_pct, paint_pct, parts_pct = _COST_SPLIT[damage_type]
    labour = round(total * labour_pct / 10) * 10
    paint = round(total * paint_pct / 10) * 10
    parts = max(0, total - labour - paint)
    
    # Updated to use the new domain.py schema
    return CostBreakdown(
        labour_cost=float(labour), 
        paint_cost=float(paint), 
        parts_cost=float(parts),
        currency="USD"
    )


def estimate(detection: Detection, view_angle: ViewAngle) -> Estimate:
    """Return a detailed repair estimate for one detection, utilizing perspective."""
    low, high, complexity, driveable, note, parts = _REPAIR_GUIDANCE[detection.damage_type]
    severity = severity_from_detection(detection)
    
    area_factor = 0.7 + min(detection.area_ratio / 0.08, 1.0) * 0.5
    confidence_factor = 0.85 + min(max(detection.confidence, 0.0), 1.0) * 0.15
    severity_multiplier = {Severity.MINOR: 0.85, Severity.MODERATE: 1.0, Severity.SEVERE: 1.25}[severity]

    low_cost = round(low * area_factor * confidence_factor * severity_multiplier / 10) * 10
    high_cost = round(high * area_factor * confidence_factor * severity_multiplier / 10) * 10
    mid_total = (low_cost + high_cost) // 2

    priority = priority_from(detection, severity, driveable)
    shop_low, shop_high = shop_time_for(detection.damage_type, severity)

    return Estimate(
        low_cost=float(low_cost),
        high_cost=float(high_cost),
        complexity=complexity,
        driveable=driveable,
        safety_note=note,
        parts=parts,
        severity=severity,
        priority=priority,
        # Pass view_angle here so the heuristic engine can resolve the part location
        vehicle_part=locate_vehicle_part(detection, view_angle),
        repair_steps=repair_steps_for(detection.damage_type, severity),
        shop_time_low_hours=shop_low,
        shop_time_high_hours=shop_high,
        cost_breakdown=_cost_breakdown(detection.damage_type, int(mid_total)),
        currency="USD"
    )
