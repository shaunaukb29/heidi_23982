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


class UltralyticsDamageDetector:
    """Adapter for a YOLO model trained on the six CarDD damage classes."""

    label_aliases = {
        "dent": DamageType.DENT, "scratch": DamageType.SCRATCH, "crack": DamageType.CRACK,
        "glass shatter": DamageType.GLASS_SHATTER, "glass_shatter": DamageType.GLASS_SHATTER,
        "shattered glass": DamageType.GLASS_SHATTER, "lamp broken": DamageType.LAMP_BROKEN,
        "lamp_broken": DamageType.LAMP_BROKEN, "broken lamp": DamageType.LAMP_BROKEN,
        "tire flat": DamageType.TIRE_FLAT, "tire_flat": DamageType.TIRE_FLAT,
        "flat tire": DamageType.TIRE_FLAT,
    }

    def __init__(self, model_path: Path, device: str) -> None:
        if not model_path.is_file():
            raise DetectorUnavailable(
                f"YOLO damage-model weights are missing at {model_path}. Add a CarDD-trained .pt file there."
            )
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise DetectorUnavailable("Install the Ultralytics detector dependency.") from error
        self._model = YOLO(str(model_path))
        self._device = device

    def inspect(self, image: Image.Image) -> list[Detection]:
        rgb = image.convert("RGB")
        result = self._model.predict(np.asarray(rgb), device=self._device, verbose=False)[0]
        if result.boxes is None:
            return []
        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        detections: list[Detection] = []
        for index, box in enumerate(result.boxes):
            confidence = float(box.conf.item())
            if confidence < 0.5:
                continue
            label = str(result.names[int(box.cls.item())]).lower().replace("-", " ")
            damage_type = self.label_aliases.get(label)
            if damage_type is None:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            mask = None
            if masks is not None:
                mask = np.asarray(masks[index], dtype=bool)
                if mask.shape != (rgb.height, rgb.width):
                    mask = np.asarray(
                        Image.fromarray(mask).resize(
                            (rgb.width, rgb.height), Image.Resampling.NEAREST
                        ),
                        dtype=bool,
                    )
            detections.append(Detection(damage_type, confidence, (float(x1), float(y1), float(x2), float(y2)), rgb.width, rgb.height, mask=mask))
        return detections


def load_detector() -> DamageDetector:
    root = Path(os.getenv("CARITHM_MODEL_DIR", "models"))
    backend = os.getenv("CARITHM_DETECTOR", "ultralytics").lower()
    device = os.getenv("CARITHM_DEVICE", "cpu")
    if backend == "ultralytics":
        return UltralyticsDamageDetector(Path(os.getenv("CARITHM_YOLO_MODEL", root / "damage-yolo.pt")), device)
    if backend != "mmdetection":
        raise DetectorUnavailable("CARITHM_DETECTOR must be 'ultralytics' or 'mmdetection'.")

    return MMDetectionDamageDetector(
        root / "dcn_plus_cfg_small.py",
        root / "checkpoint.pth",
        device,
    )
