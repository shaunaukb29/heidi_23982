from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DamageType(str, Enum):
    DENT = "dent"
    SCRATCH = "scratch"
    CRACK = "crack"
    GLASS_SHATTER = "glass_shatter"
    LAMP_BROKEN = "lamp_broken"
    TIRE_FLAT = "tire_flat"


@dataclass(frozen=True)
class Detection:
    damage_type: DamageType
    confidence: float
    bbox: tuple[float, float, float, float]
    image_width: int
    image_height: int

    @property
    def area_ratio(self) -> float:
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
