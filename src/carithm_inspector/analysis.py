from __future__ import annotations

from .domain import DamageType, Detection, Priority, Severity, VehiclePart


def _side(cx: float) -> str:
    if cx < 0.35:
        return "Left"
    if cx > 0.65:
        return "Right"
    return "Center"


def locate_vehicle_part(detection: Detection) -> VehiclePart:
    """Estimate affected panel from bbox position and damage type."""
    x1, y1, x2, y2 = detection.bbox
    w, h = detection.image_width, detection.image_height
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    side = _side(cx)
    damage = detection.damage_type

    if damage == DamageType.TIRE_FLAT:
        return VehiclePart(f"{side} Wheel", "Tire and wheel assembly", "Rubber tire, alloy wheel")

    if damage == DamageType.LAMP_BROKEN:
        if cy < 0.45:
            loc = f"{side} Front Headlight".replace("Center ", "")
        else:
            loc = f"{side} Rear Taillight".replace("Center ", "")
        return VehiclePart(loc, "Lamp assembly", "Polycarbonate lens, housing")

    if damage == DamageType.GLASS_SHATTER:
        if cy < 0.35:
            return VehiclePart("Windshield", "Laminated safety glass", "Glass")
        if cx < 0.35:
            return VehiclePart("Left Side Window", "Tempered side glass", "Glass")
        if cx > 0.65:
            return VehiclePart("Right Side Window", "Tempered side glass", "Glass")
        return VehiclePart("Rear Window", "Tempered rear glass", "Glass")

    if cy < 0.25:
        if side != "Center":
            return VehiclePart(f"{side} Hood", "Steel hood panel", "Steel")
        return VehiclePart("Hood or Roof", "Steel body panel", "Steel")

    if cy < 0.45:
        if side != "Center":
            return VehiclePart(f"{side} Front Fender", "Steel fender panel", "Steel")
        return VehiclePart("Front Grille Area", "Composite grille surround", "Plastic/composite")

    if cy < 0.65:
        if side != "Center":
            return VehiclePart(f"{side} Front Door", "Steel door panel", "Steel")
        return VehiclePart("Center Body Panel", "Steel body panel", "Steel")

    if cy < 0.80:
        if side != "Center":
            return VehiclePart(f"{side} Rear Door", "Steel door panel", "Steel")
        return VehiclePart("Trunk or Liftgate", "Steel rear panel", "Steel")

    if side != "Center":
        label = f"{side} Front Bumper" if cy > 0.88 else f"{side} Side Skirt"
        panel = "Plastic bumper cover" if cy > 0.88 else "Plastic rocker cover"
        return VehiclePart(label, panel, "Plastic")

    return VehiclePart("Front Bumper", "Plastic bumper cover", "Plastic")


def severity_from_detection(detection: Detection) -> Severity:
    """Classify severity from visible surface area."""
    area = detection.area_ratio
    thresholds = {
        DamageType.SCRATCH: (0.008, 0.035),
        DamageType.DENT: (0.012, 0.055),
        DamageType.CRACK: (0.006, 0.025),
        DamageType.GLASS_SHATTER: (0.010, 0.040),
        DamageType.LAMP_BROKEN: (0.005, 0.020),
        DamageType.TIRE_FLAT: (0.015, 0.050),
    }
    minor_max, moderate_max = thresholds[detection.damage_type]
    if area < minor_max:
        return Severity.MINOR
    if area < moderate_max:
        return Severity.MODERATE
    return Severity.SEVERE


def priority_from(detection: Detection, severity: Severity, driveable: bool) -> Priority:
    if not driveable:
        return Priority.SAFETY_CRITICAL

    critical_types = {DamageType.GLASS_SHATTER, DamageType.TIRE_FLAT}
    if detection.damage_type in critical_types:
        return Priority.SAFETY_CRITICAL

    if detection.damage_type in {DamageType.CRACK, DamageType.LAMP_BROKEN}:
        if severity == Severity.MINOR:
            return Priority.MODERATE
        return Priority.SAFETY_CRITICAL

    if severity == Severity.SEVERE:
        return Priority.MODERATE

    if severity == Severity.MODERATE:
        return Priority.MODERATE

    return Priority.COSMETIC


def repair_steps_for(damage_type: DamageType, severity: Severity) -> tuple[str, ...]:
    guidance: dict[DamageType, dict[Severity, tuple[str, ...]]] = {
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
    return guidance[damage_type][severity]


def shop_time_for(damage_type: DamageType, severity: Severity) -> tuple[float, float]:
    hours: dict[DamageType, dict[Severity, tuple[float, float]]] = {
        DamageType.SCRATCH: {
            Severity.MINOR: (1.0, 2.0),
            Severity.MODERATE: (2.0, 4.0),
            Severity.SEVERE: (4.0, 8.0),
        },
        DamageType.DENT: {
            Severity.MINOR: (1.5, 3.0),
            Severity.MODERATE: (3.0, 6.0),
            Severity.SEVERE: (6.0, 12.0),
        },
        DamageType.CRACK: {
            Severity.MINOR: (2.0, 4.0),
            Severity.MODERATE: (4.0, 8.0),
            Severity.SEVERE: (8.0, 16.0),
        },
        DamageType.GLASS_SHATTER: {
            Severity.MINOR: (1.0, 2.0),
            Severity.MODERATE: (2.0, 4.0),
            Severity.SEVERE: (3.0, 6.0),
        },
        DamageType.LAMP_BROKEN: {
            Severity.MINOR: (0.5, 1.5),
            Severity.MODERATE: (1.5, 3.0),
            Severity.SEVERE: (2.0, 5.0),
        },
        DamageType.TIRE_FLAT: {
            Severity.MINOR: (0.5, 1.0),
            Severity.MODERATE: (1.0, 2.0),
            Severity.SEVERE: (2.0, 4.0),
        },
    }
    return hours[damage_type][severity]
