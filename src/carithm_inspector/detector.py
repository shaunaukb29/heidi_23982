from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .domain import DamageType, Detection


class DetectorUnavailable(RuntimeError):
    pass


class DamageDetector(Protocol):
    def inspect(self, image: Image.Image) -> list[Detection]:
        ...


class MMDetectionDamageDetector:
    """Thin adapter for the CarDD MMDetection checkpoint used by AutoInspector."""

    labels = {
        1: DamageType.DENT,
        2: DamageType.SCRATCH,
        3: DamageType.CRACK,
        4: DamageType.GLASS_SHATTER,
        5: DamageType.LAMP_BROKEN,
        6: DamageType.TIRE_FLAT,
    }

    def __init__(self, config_path: Path, checkpoint_path: Path, device: str) -> None:
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise DetectorUnavailable("Detector config or checkpoint is not installed.")

        try:
            from mmdet.apis import inference_detector, init_detector
        except ImportError as error:
            raise DetectorUnavailable(
                f"MMDetection could not import: {error}"
            ) from error

        self._infer = inference_detector

        if device.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    device = "cpu"
            except Exception:
                device = "cpu"

        self._model = init_detector(
            str(config_path),
            str(checkpoint_path),
            device=device,
        )

    def inspect(self, image: Image.Image) -> list[Detection]:
        rgb = image.convert("RGB")
        result = self._infer(self._model, np.asarray(rgb))

        if isinstance(result, tuple):
            boxes_by_class, masks_by_class = result
        else:
            boxes_by_class = result
            masks_by_class = None

        detections: list[Detection] = []

        for class_index, boxes in enumerate(boxes_by_class, start=1):
            damage_type = self.labels.get(class_index)
            if damage_type is None:
                continue

            class_masks = None
            if masks_by_class is not None and class_index - 1 < len(masks_by_class):
                class_masks = masks_by_class[class_index - 1]

            for box_index, (x1, y1, x2, y2, score) in enumerate(boxes):
                if float(score) < 0.5:
                    continue

                mask = None
                if class_masks is not None and box_index < len(class_masks):
                    raw_mask = class_masks[box_index]
                    if raw_mask is not None:
                        mask = np.asarray(raw_mask, dtype=bool)

                detections.append(
                    Detection(
                        damage_type,
                        float(score),
                        (float(x1), float(y1), float(x2), float(y2)),
                        rgb.width,
                        rgb.height,
                        mask=mask,
                    )
                )

        return detections


def load_detector() -> DamageDetector:
    root = Path(os.getenv("CARITHM_MODEL_DIR", "models"))

    return MMDetectionDamageDetector(
        root / "dcn_plus_cfg_small.py",
        root / "checkpoint.pth",
        os.getenv("CARITHM_DEVICE", "cpu"),
    )
