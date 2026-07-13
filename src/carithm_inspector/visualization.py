from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .domain import DamageType, Detection, Priority, Severity


DAMAGE_ICONS: dict[DamageType, str] = {
    DamageType.SCRATCH: "🟡",
    DamageType.DENT: "🟠",
    DamageType.CRACK: "🔴",
    DamageType.GLASS_SHATTER: "🟣",
    DamageType.LAMP_BROKEN: "🟠",
    DamageType.TIRE_FLAT: "🟢",
}

DAMAGE_COLORS: dict[DamageType, tuple[int, int, int]] = {
    DamageType.SCRATCH: (255, 193, 7),
    DamageType.DENT: (33, 150, 243),
    DamageType.CRACK: (244, 67, 54),
    DamageType.GLASS_SHATTER: (156, 39, 176),
    DamageType.LAMP_BROKEN: (255, 87, 34),
    DamageType.TIRE_FLAT: (76, 175, 80),
}

SEVERITY_ICONS = {
    Severity.MINOR: "🟢",
    Severity.MODERATE: "🟠",
    Severity.SEVERE: "🔴",
}

PRIORITY_LABELS = {
    Priority.COSMETIC: ("🟢 Cosmetic", "No urgent repair required."),
    Priority.MODERATE: ("🟠 Moderate", "Repair within 30 days."),
    Priority.SAFETY_CRITICAL: ("🔴 Safety Critical", "Driving not recommended."),
}


@dataclass(frozen=True)
class OverlayStyle:
    mask_alpha: int = 90
    bbox_width: int = 3
    selected_bbox_width: int = 5
    label_bg_alpha: int = 180


def _damage_label(detection: Detection, index: int) -> str:
    icon = DAMAGE_ICONS[detection.damage_type]
    name = detection.damage_type.replace("_", " ").title()
    return f"{icon} {name} {detection.confidence:.0%}"


def render_inspection_overlay(
    image: Image.Image,
    detections: list[Detection],
    *,
    selected_index: int | None = None,
    style: OverlayStyle | None = None,
) -> Image.Image:
    """Draw bounding boxes and segmentation masks on the inspection photo."""
    style = style or OverlayStyle()
    base = image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for index, detection in enumerate(detections):
        color = DAMAGE_COLORS[detection.damage_type]
        is_selected = selected_index == index
        alpha = style.mask_alpha + (40 if is_selected else 0)
        width = style.selected_bbox_width if is_selected else style.bbox_width

        if detection.mask is not None and detection.mask.any():
            mask_rgba = np.zeros((*detection.mask.shape, 4), dtype=np.uint8)
            mask_rgba[detection.mask] = (*color, alpha)
            mask_image = Image.fromarray(mask_rgba, mode="RGBA")
            overlay = Image.alpha_composite(overlay, mask_image)
        else:
            x1, y1, x2, y2 = detection.bbox
            draw.rectangle(
                [x1, y1, x2, y2],
                fill=(*color, alpha // 2),
                outline=(*color, 255),
                width=width,
            )

        x1, y1, x2, y2 = detection.bbox
        draw.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=width)

        label = _damage_label(detection, index)
        text_bbox = draw.textbbox((0, 0), label)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_y = max(0, y1 - text_h - 8)
        draw.rectangle(
            [x1, label_y, x1 + text_w + 8, label_y + text_h + 6],
            fill=(*color, style.label_bg_alpha),
        )
        draw.text((x1 + 4, label_y + 2), label, fill=(255, 255, 255, 255))

    composed = Image.alpha_composite(base, overlay)
    return composed.convert("RGB")


def format_damage_type(damage_type: DamageType) -> str:
    return damage_type.replace("_", " ").title()
