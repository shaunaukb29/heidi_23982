# Carithm Visual Inspector

Stateless API for vehicle-photo damage detection and a cautious repair-range estimate. It has no accounts, database, maps, templates, or repair-shop portal.

The API accepts a JPEG, PNG, or WebP image and returns detected CarDD classes: dent, scratch, crack, shattered glass, broken lamp, and flat tire. Each detection includes a bounding box, model confidence, repair range, complexity, common parts, and a safety flag.

## Run it

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[detector,dev]"
carithm-inspector
```

Set `CARITHM_MODEL_DIR` if the config and checkpoint live elsewhere. Set `CARITHM_DEVICE=cpu` for CPU inference.

The compatible configuration is included in `models/dcn_plus_cfg_small.py`. The 490 MB checkpoint is intentionally ignored by Git and must be provided locally as `models/checkpoint.pth`. Until it is present, `/health` returns `degraded` and the inspection endpoint returns `503`; it never returns invented detections.

## API

```bash
curl -F "image=@vehicle.jpg" http://localhost:8000/v1/inspections
```

```json
{
  "damages": [{
    "type": "scratch",
    "confidence": 0.84,
    "bounding_box": [120.0, 220.0, 510.0, 340.0],
    "estimate": {
      "low_usd": 180,
      "high_usd": 350,
      "complexity": "low",
      "driveable": true,
      "safety_note": "Cosmetic damage can hide sharp edges.",
      "common_parts": ["paint", "clear coat"]
    }
  }],
  "disclaimer": "Image-only estimate. It cannot confirm hidden, structural, or mechanical damage. Obtain an in-person inspection for safety decisions."
}
```

Repair figures are broad US consumer ranges, not quotes. The visible area and detection confidence adjust the range. Integrate this service behind the Carithm frontend; do not use a photo result as a safety clearance.

## Streamlit app

```bash
python -m pip install -e ".[app]"
streamlit run streamlit_app.py
```

The app uses the same local detector as the API. If the detector is not installed, it shows a clear unavailable state instead of inventing a result.

## Run on Apple Silicon

This checkpoint depends on legacy MMCV operators. Native compilation currently fails against recent macOS SDKs, so run the supported Linux CPU image instead. Docker Desktop is required.

```bash
docker build --platform linux/amd64 -f Dockerfile.cpu -t carithm-visual-inspector .
docker run --rm --platform linux/amd64 -p 8000:8000 carithm-visual-inspector
```

The x86 image runs under emulation on Apple Silicon and is suitable for local testing, not high-throughput inference.

## Tests

```bash
pytest
```

## Source credit

The MMDetection adapter follows the six-class mapping and checkpoint layout documented by [A.I.-AutoInspector](https://github.com/Divyeshpratap/A.I.-AutoInspector), licensed MIT. This project is a clean-room, API-only extraction; it does not include that application's Flask UI, accounts, database, portal, map integration, or manual RAG.
