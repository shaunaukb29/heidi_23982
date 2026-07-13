from __future__ import annotations

import logging
from enum import Enum
from typing import Mapping, Tuple

# Assuming these are defined in your .domain module
from .domain import DamageType, Detection, Priority, Severity, VehiclePart

logger = logging.getLogger(__name__)


class ViewAngle(str, Enum):
    """Explicitly defines the camera's perspective of the vehicle."""
    FRONT = "Front"
    REAR = "Rear"
    LEFT_SIDE = "Left_Side"
    RIGHT_SIDE = "Right_Side"
    UNKNOWN = "Unknown"


# ==========================================
# MODULE-LEVEL CONFIGURATION REGISTRIES
# ==========================================
# Extracted from functions for O(1) lookups and memory efficiency.

SEVERITY_THRESHOLDS: Mapping[DamageType, Tuple[float, float]] = {
    DamageType.SCRATCH: (0.008, 0.035),
    DamageType.DENT: (0.012, 0.055),
    DamageType.CRACK: (0.006, 0.025),
    DamageType.GLASS_SHATTER: (0.010, 0.040),
    DamageType.LAMP_BROKEN: (0.005, 0.020),
    DamageType.TIRE_FLAT: (0.015, 0.050),
}

REPAIR_GUIDANCE: Mapping[DamageType, Mapping[Severity, Tuple[str, ...]]] = {
    DamageType.SCRATCH: {
        Severity.MINOR: ("Buffing", "Paint correction"),
        Severity.MODERATE: ("Wet sanding", "Spot paint blend"),
        Severity.SEVERE: ("Panel refinish", "Clear coat application"),
    },
    DamageType.DENT: {
        Severity.MINOR: ("Paintless dent repair (PDR)", "Light polish"),
        Severity.MODERATE: ("Body filler", "Paint blend"),
        Severity.SEVERE: ("Panel reshaping or replacement", "Full panel refinish"),
    },
    DamageType.CRACK: {
        Severity.MINOR: ("Crack seal and blend", "Structural inspection"),
        Severity.MODERATE: ("Panel section repair", "Reinforcement check"),
        Severity.SEVERE: ("Panel replacement", "Mount point inspection"),
    },
    DamageType.GLASS_SHATTER: {
        Severity.MINOR: ("Chip repair assessment",),
        Severity.MODERATE: ("Glass replacement", "Seal replacement"),
        Severity.SEVERE: ("Full glass assembly replacement", "Frame inspection"),
    },
    DamageType.LAMP_BROKEN: {
        Severity.MINOR: ("Lens polish or seal check",),
        Severity.MODERATE: ("Lamp assembly replacement", "Wiring inspection"),
        Severity.SEVERE: ("Full lamp assembly replacement", "Electrical harness check"),
    },
    DamageType.TIRE_FLAT: {
        Severity.MINOR: ("Tire plug or patch", "Pressure check"),
        Severity.MODERATE: ("Tire replacement", "Wheel balance"),
        Severity.SEVERE: ("Tire and wheel replacement", "Suspension inspection"),
    },
}

SHOP_TIME_ESTIMATES: Mapping[DamageType, Mapping[Severity, Tuple[float, float]]] = {
    DamageType.SCRATCH: {Severity.MINOR: (1.0, 2.0), Severity.MODERATE: (2.0, 4.0), Severity.SEVERE: (4.0, 8.0)},
    DamageType.DENT: {Severity.MINOR: (1.5, 3.0), Severity.MODERATE: (3.0, 6.0), Severity.SEVERE: (6.0, 12.0)},
    DamageType.CRACK: {Severity.MINOR: (2.0, 4.0), Severity.MODERATE: (4.0, 8.0), Severity.SEVERE: (8.0, 16.0)},
    DamageType.GLASS_SHATTER: {Severity.MINOR: (1.0, 2.0), Severity.MODERATE: (2.0, 4.0), Severity.SEVERE: (3.0, 6.0)},
    DamageType.LAMP_BROKEN: {Severity.MINOR: (0.5, 1.5), Severity.MODERATE: (1.5, 3.0), Severity.SEVERE: (2.0, 5.0)},
    DamageType.TIRE_FLAT: {Severity.MINOR: (0.5, 1.0), Severity.MODERATE: (1.0, 2.0), Severity.SEVERE: (2.0, 4.0)},
}


# ==========================================
# CORE HEURISTIC ENGINES
# ==========================================

def _get_horizontal_zone(cx: float) -> str:
    """Resolve normalized X coordinate to a semantic horizontal zone."""
    if cx < 0.20: return "Far Left"
    if cx < 0.40: return "Left"
    if cx < 0.60: return "Center"
    if cx < 0.80: return "Right"
    return "Far Right"


def locate_vehicle_part(
    detection: Detection, 
    view_angle: ViewAngle = ViewAngle.UNKNOWN
) -> VehiclePart:
    """
    Estimate the affected vehicle component using spatial bounding boxes and camera perspective.
    """
    x1, y1, x2, y2 = detection.bbox
    cx = ((x1 + x2) / 2) / detection.image_width
    cy = ((y1 + y2) / 2) / detection.image_height
    damage = detection.damage_type

    zone = _get_horizontal_zone(cx)
    is_left = "Left" in zone
    is_right = "Right" in zone

    # 1. Handle explicit components tied to damage types
    if damage == DamageType.TIRE_FLAT:
        side_modifier = zone.replace("Far ", "")
        return VehiclePart(f"{side_modifier} Wheel", "Tire and wheel assembly", "Rubber / Alloy")

    if damage == DamageType.LAMP_BROKEN:
        side_modifier = zone.replace("Center ", "").replace("Far ", "")
        lamp_type = "Headlight" if view_angle == ViewAngle.FRONT or cy < 0.45 else "Tail Light"
        return VehiclePart(f"{side_modifier} {lamp_type}".strip(), "Lamp assembly", "Polycarbonate")

    if damage == DamageType.GLASS_SHATTER:
        if cy < 0.28 or view_angle == ViewAngle.FRONT:
            return VehiclePart("Windshield", "Front laminated glass", "Glass")
        if cy > 0.72 or view_angle == ViewAngle.REAR:
            return VehiclePart("Rear Window", "Rear safety glass", "Glass")
        side_label = "Left" if cx < 0.5 else "Right"
        return VehiclePart(f"{side_label} Side Window", "Tempered glass", "Glass")

    # 2. Vertical Spatial Tree (General Body Panels)
    if cy < 0.12:
        return VehiclePart("Roof", "Roof panel", "Steel")

    if cy < 0.28:
        hood_label = "Left Hood Edge" if is_left else ("Right Hood Edge" if is_right else "Hood")
        return VehiclePart(hood_label, "Steel hood panel", "Steel")

    if cy < 0.45:
        if is_left: return VehiclePart("Left Front Fender", "Steel fender", "Steel")
        if is_right: return VehiclePart("Right Front Fender", "Steel fender", "Steel")
        return VehiclePart("Front Grille", "Grille assembly", "Plastic / Composite")

    if cy < 0.70:
        # Resolve doors safely, acknowledging we might be looking at the right side of the car
        if view_angle == ViewAngle.RIGHT_SIDE:
            if zone in ("Far Left", "Left"): return VehiclePart("Right Rear Door", "Steel door panel", "Steel")
            if zone in ("Far Right", "Right"): return VehiclePart("Right Front Door", "Steel door panel", "Steel")
        else:  # Default left-facing assumption
            if zone in ("Far Left", "Left"): return VehiclePart("Left Front Door", "Steel door panel", "Steel")
            if zone in ("Far Right", "Right"): return VehiclePart("Left Rear Door", "Steel door panel", "Steel")
        return VehiclePart("Center Pillar", "Body structure", "Steel")

    if cy < 0.86:
        if is_left: return VehiclePart("Left Rear Quarter Panel", "Steel body panel", "Steel")
        if is_right: return VehiclePart("Right Rear Quarter Panel", "Steel body panel", "Steel")
        return VehiclePart("Trunk / Liftgate", "Rear body panel", "Steel")

    # 3. Bottom Layer (Bumpers) - Perspective-Aware
    is_front = view_angle == ViewAngle.FRONT
    prefix = "Front" if is_front else "Rear"
    
    if is_left: return VehiclePart(f"{prefix} Left Bumper", "Plastic bumper cover", "Plastic")
    if is_right: return VehiclePart(f"{prefix} Right Bumper", "Plastic bumper cover", "Plastic")
    return VehiclePart(f"{prefix} Bumper", "Plastic bumper cover", "Plastic")


def severity_from_detection(detection: Detection) -> Severity:
    """Classify severity from visible surface area with safe fallback thresholds."""
    # Default to a generic moderate threshold if damage type is unknown
    minor_max, moderate_max = SEVERITY_THRESHOLDS.get(
        detection.damage_type, (0.010, 0.040)
    )
    
    if detection.damage_type not in SEVERITY_THRESHOLDS:
        logger.warning(f"Unmapped damage type '{detection.damage_type}'. Defaulting thresholds.")

    area = detection.area_ratio
    if area < minor_max:
        return Severity.MINOR
    if area < moderate_max:
        return Severity.MODERATE
    return Severity.SEVERE


def priority_from(detection: Detection, severity: Severity, driveable: bool) -> Priority:
    """Determine repair priority based on safety and drivability."""
    if not driveable:
        return Priority.SAFETY_CRITICAL

    critical_types = frozenset({DamageType.GLASS_SHATTER, DamageType.TIRE_FLAT})
    if detection.damage_type in critical_types:
        return Priority.SAFETY_CRITICAL

    moderate_to_critical_types = frozenset({DamageType.CRACK, DamageType.LAMP_BROKEN})
    if detection.damage_type in moderate_to_critical_types:
        return Priority.MODERATE if severity == Severity.MINOR else Priority.SAFETY_CRITICAL

    if severity in (Severity.SEVERE, Severity.MODERATE):
        return Priority.MODERATE

    return Priority.COSMETIC


def repair_steps_for(damage_type: DamageType, severity: Severity) -> Tuple[str, ...]:
    """Retrieve repair steps with fail-safe dictionary lookups."""
    damage_map = REPAIR_GUIDANCE.get(damage_type)
    if not damage_map:
        logger.error(f"No repair guidance mapped for {damage_type}.")
        return ("Diagnostic assessment required",)

    return damage_map.get(severity, ("Custom repair plan required",))


def shop_time_for(damage_type: DamageType, severity: Severity) -> Tuple[float, float]:
    """Retrieve estimated shop time with fail-safe dictionary lookups."""
    time_map = SHOP_TIME_ESTIMATES.get(damage_type)
    if not time_map:
        logger.error(f"No shop time mapped for {damage_type}.")
        return (0.0, 0.0)

    return time_map.get(severity, (0.0, 0.0))
