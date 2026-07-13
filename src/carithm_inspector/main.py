from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from .detector import DamageDetector, DetectorUnavailable, load_detector
from .estimates import estimate


class EstimateResponse(BaseModel):
    low_usd: int
    high_usd: int
    complexity: str
    driveable: bool
    safety_note: str
    common_parts: list[str]


class DamageResponse(BaseModel):
    type: str
    confidence: float
    bounding_box: list[float]
    estimate: EstimateResponse


class InspectionResponse(BaseModel):
    damages: list[DamageResponse]
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
        yield

    app = FastAPI(title="Carithm Visual Inspector", version="0.1.0", lifespan=lifespan)

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
        damages = []
        for detection in app.state.detector.inspect(photo):
            repair = estimate(detection)
            damages.append(DamageResponse(
                type=detection.damage_type,
                confidence=round(detection.confidence, 3),
                bounding_box=[round(value, 1) for value in detection.bbox],
                estimate=EstimateResponse(**repair.__dict__, common_parts=list(repair.parts)),
            ))
        return InspectionResponse(
            damages=damages,
            disclaimer="Image-only estimate. It cannot confirm hidden, structural, or mechanical damage. Obtain an in-person inspection for safety decisions.",
        )

    return app


app = create_app()


def run() -> None:
    import uvicorn
    uvicorn.run("carithm_inspector.main:app", host="0.0.0.0", port=8000, reload=True)
