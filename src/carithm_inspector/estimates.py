from __future__ import annotations

from .domain import DamageType, Detection, Estimate


# Baselines are consumer repair ranges in USD, before visible-area adjustment.
_REPAIR_GUIDANCE: dict[DamageType, tuple[int, int, str, bool, str, tuple[str, ...]]] = {
    DamageType.SCRATCH: (180, 350, "low", True, "Cosmetic damage can hide sharp edges.", ("paint", "clear coat")),
    DamageType.DENT: (200, 650, "medium", True, "Check that panels and doors still operate normally.", ("body panel", "paint")),
    DamageType.CRACK: (250, 900, "medium", False, "A crack near safety glass, lights, or a bumper mount needs inspection before driving.", ("body panel", "bumper cover")),
    DamageType.GLASS_SHATTER: (300, 1_200, "high", False, "Broken vehicle glass can obstruct vision and leave loose shards.", ("glass assembly", "weather seal")),
    DamageType.LAMP_BROKEN: (180, 1_100, "medium", False, "Do not drive at night or in poor visibility with a damaged lamp.", ("lamp assembly", "bulb or LED module")),
    DamageType.TIRE_FLAT: (120, 450, "medium", False, "Do not drive on a flat tire. Fit the spare or arrange recovery.", ("tire", "valve stem", "wheel repair")),
}


def estimate(detection: Detection) -> Estimate:
    """Return a cautious, visible-damage-only repair range for one detection."""
    low, high, complexity, driveable, note, parts = _REPAIR_GUIDANCE[detection.damage_type]
    # Very small boxes are often chips, while a large box tends to require more prep or replacement.
    severity = 0.7 + min(detection.area_ratio / 0.08, 1.0) * 0.5
    confidence = 0.85 + min(max(detection.confidence, 0.0), 1.0) * 0.15
    return Estimate(
        low_usd=round(low * severity * confidence / 10) * 10,
        high_usd=round(high * severity * confidence / 10) * 10,
        complexity=complexity,
        driveable=driveable,
        safety_note=note,
        parts=parts,
    )
