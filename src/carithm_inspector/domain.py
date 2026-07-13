from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class DamageType(str, Enum):
    DENT = "dent"
    SCRATCH = "scratch"
    CRACK = "crack"
    GLASS_SHATTER = "glass_shatter"
    LAMP_BROKEN = "lamp_broken"
    TIRE_FLAT = "tire_flat"


class Severity(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class Priority(str, Enum):
    COSMETIC = "cosmetic"
    MODERATE = "moderate"
    SAFETY_CRITICAL = "safety_critical"


@dataclass(frozen=True)
class VehiclePart:
    location: str
    panel: str
    material: str


@dataclass(frozen=True)
class CostBreakdown:
    labour_usd: int
    paint_usd: int
    parts_usd: int

    @property
    def total_usd(self) -> int:
        return self.labour_usd + self.paint_usd + self.parts_usd


@dataclass(frozen=True)
class Detection:
    damage_type: DamageType
    confidence: float
    bbox: tuple[float, float, float, float]
    image_width: int
    image_height: int
    mask: np.ndarray | None = None

    @property
    def area_ratio(self) -> float:
        if self.mask is not None and self.mask.any():
            return min(1.0, float(self.mask.sum()) / (self.image_width * self.image_height))
        x1, y1, x2, y2 = self.bbox
        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return min(1.0, box_area / (self.image_width * self.image_height))


@dataclass(frozen=True)
class Estimate:
    low_usd: int
    high_usd: int
    complexity: str
    driveable: bool
    safety_note: str
    parts: tuple[str, ...]
    severity: Severity
    priority: Priority
    vehicle_part: VehiclePart
    repair_steps: tuple[str, ...]
    shop_time_low_hours: float
    shop_time_high_hours: float
    cost_breakdown: CostBreakdown
