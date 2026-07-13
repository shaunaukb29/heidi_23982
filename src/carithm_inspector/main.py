from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from .detector import DamageDetector, DetectorUnavailable, load_detector
from .estimates import estimate
from .temporary_models import HeuristicOrientationModel, NullPartSegmenter


class VehiclePartResponse(BaseModel):
    location: str
    panel: str
    material: str


class CostBreakdownResponse(BaseModel):
    labour_usd: int
    paint_usd: int
    parts_usd: int
    total_usd: int


class EstimateResponse(BaseModel):
    low_usd: int
    high_usd: int
    complexity: str
    driveable: bool
    safety_note: str
    common_parts: list[str]
    severity: str
    priority: str
    vehicle_part: VehiclePartResponse
    repair_steps: list[str]
    shop_time_low_hours: float
    shop_time_high_hours: float
    cost_breakdown: CostBreakdownResponse


class DamageResponse(BaseModel):
    type: str
    confidence: float
    bounding_box: list[float]
    surface_area_pct: float
    has_segmentation_mask: bool
    estimate: EstimateResponse


class InspectionResponse(BaseModel):
    damages: list[DamageResponse]
    damage_count: int
    disclaimer: str


def create_app(detector: DamageDetector | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if detector is not None:
            app.state.detector = detector
            app.state.detector_error = None
        else:
            try:
                app.state.detector = load_detector()
                app.state.detector_error = None
            except DetectorUnavailable as error:
                app.state.detector = None
                app.state.detector_error = str(error)

        # Temporary stand-ins until trained OrientationModel / PartSegmenter
        # heads exist. NullPartSegmenter -> estimate() takes the legacy
        # bbox-heuristic fallback path automatically (see part_mapping.py).
        app.state.orientation_model = HeuristicOrientationModel()
        app.state.part_segmenter = NullPartSegmenter()
        yield

    app = FastAPI(title="Carithm Visual Inspector", version="0.2.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok" if app.state.detector else "degraded", "detector": app.state.detector_error or "ready"}

    @app.post("/v1/inspections", response_model=InspectionResponse)
    async def inspect_vehicle(image: UploadFile = File(...)) -> InspectionResponse:
        if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(415, "Upload a JPEG, PNG, or WebP image.")
        if app.state.detector is None:
            raise HTTPException(503, app.state.detector_error)
        payload = await image.read()
        if len(payload) > 12 * 1024 * 1024:
            raise HTTPException(413, "Image must be 12 MB or smaller.")
        try:
            photo = Image.open(BytesIO(payload))
            photo.verify()
            photo = Image.open(BytesIO(payload))
        except UnidentifiedImageError as error:
            raise HTTPException(422, "The uploaded file is not a valid image.") from error

        view_angle = app.state.orientation_model.predict(photo)
        part_mask = app.state.part_segmenter.predict(photo)

        damages = []
        for detection in app.state.detector.inspect(photo):
            repair = estimate(detection, view_angle, part_mask)
            breakdown = repair.cost_breakdown
            damages.append(
                DamageResponse(
                    type=detection.damage_type,
                    confidence=round(detection.confidence, 3),
                    bounding_box=[round(value, 1) for value in detection.bbox],
                    surface_area_pct=round(detection.area_ratio * 100, 2),
                    has_segmentation_mask=detection.mask is not None and detection.mask.any(),
                    estimate=EstimateResponse(
                        low_usd=int(repair.low_cost),
                        high_usd=int(repair.high_cost),
                        complexity=repair.complexity,
                        driveable=repair.driveable,
                        safety_note=repair.safety_note,
                        common_parts=list(repair.parts),
                        severity=repair.severity.value,
                        priority=repair.priority.value,
                        vehicle_part=VehiclePartResponse(
                            location=repair.vehicle_part.location,
                            panel=repair.vehicle_part.panel,
                            material=repair.vehicle_part.material,
                        ),
                        repair_steps=list(repair.repair_steps),
                        shop_time_low_hours=repair.shop_time_low_hours,
                        shop_time_high_hours=repair.shop_time_high_hours,
                        cost_breakdown=CostBreakdownResponse(
                            labour_usd=int(breakdown.labour_cost),
                            paint_usd=int(breakdown.paint_cost),
                            parts_usd=int(breakdown.parts_cost),
                            total_usd=int(breakdown.total_cost),
                        ),
                    ),
                )
            )

        return InspectionResponse(
            damages=damages,
            damage_count=len(damages),
            disclaimer=(
                "Image-only estimate. It cannot confirm hidden, structural, or mechanical damage. "
                "Obtain an in-person inspection for safety decisions."
            ),
        )

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("carithm_inspector.main:app", host="0.0.0.0", port=8000, reload=True)
