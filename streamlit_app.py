from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image

from carithm_inspector.analysis import severity_from_detection
from carithm_inspector.detector import DetectorUnavailable, load_detector
from carithm_inspector.domain import DamageType, Detection, Severity
from carithm_inspector.visualization import format_damage_type, render_inspection_overlay

SUPPORTED_DAMAGE = {DamageType.DENT, DamageType.SCRATCH}

OVERALL_SEVERITY_ICONS = {
    Severity.MINOR: "🟢",
    Severity.MODERATE: "🟡",
    Severity.SEVERE: "🔴",
}

st.set_page_config(page_title="Carithm Inspector", page_icon="🚗", layout="wide")

st.title("Carithm Visual Inspector")
st.caption("Upload a photo — we spot scratches and dents in the image. No need to label which part of the car.")


@st.cache_resource(show_spinner="Loading the damage detector…")
def detector():
    return load_detector()


def _overall_severity(detections: list[Detection]) -> Severity:
    order = {Severity.MINOR: 0, Severity.MODERATE: 1, Severity.SEVERE: 2}
    return max((severity_from_detection(detection) for detection in detections), key=order.get)


def _overall_confidence(detections: list[Detection]) -> float:
    return sum(detection.confidence for detection in detections) / len(detections)


upload = st.file_uploader("Vehicle photo", type=["jpg", "jpeg", "png", "webp"])

if upload is not None:
    photo = Image.open(BytesIO(upload.getvalue())).convert("RGB")

    if "inspection" not in st.session_state or st.session_state.get("upload_name") != upload.name:
        st.session_state.upload_name = upload.name
        st.session_state.inspection = None

    if st.button("Run AI damage assessment", type="primary"):
        try:
            detections = [
                detection
                for detection in detector().inspect(photo)
                if detection.damage_type in SUPPORTED_DAMAGE
            ]
            st.session_state.inspection = detections
        except DetectorUnavailable as error:
            st.error(f"The damage model is unavailable: {error}")
            st.info(
                "The Streamlit interface is ready. Connect a supported detector "
                "or hosted inference endpoint to enable inspection."
            )
            st.stop()

    if st.session_state.get("inspection") is None:
        st.image(photo, caption="Uploaded photo", use_container_width=True)
        st.stop()

    detections: list[Detection] = st.session_state.inspection

    if not detections:
        st.image(photo, caption="Uploaded photo", use_container_width=True)
        st.info("No scratches or dents were detected. Try a closer, well-lit photo showing the damage.")
        st.stop()

    severity = _overall_severity(detections)
    confidence = _overall_confidence(detections)
    overlay = render_inspection_overlay(photo, detections)

    st.subheader("AI Damage Assessment")

    st.markdown("**Overall Condition**")
    st.markdown(
        f"**Estimated Severity:** {OVERALL_SEVERITY_ICONS[severity]} {severity.value.title()}"
    )
    st.markdown(f"**AI Confidence:** {confidence:.0%}")
    st.caption("A visual assessment based on the uploaded image. We spot the damage — no car part labels.")

    st.markdown("**Damage Detected**")
    for detection in detections:
        st.markdown(f"✅ **{format_damage_type(detection.damage_type)}**")
        st.markdown(f"Confidence: {detection.confidence:.0%}")

    st.markdown("**Damage Visualisation**")
    st.image(overlay, use_container_width=True)
    st.caption("Coloured bounding boxes, transparent segmentation masks, and confidence labels.")

st.caption("Visual assessment only. Cannot detect hidden, structural, or mechanical damage.")
