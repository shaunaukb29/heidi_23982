from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image

from carithm_inspector.detector import DetectorUnavailable, load_detector
from carithm_inspector.estimates import estimate


st.set_page_config(page_title="Carithm Inspector", page_icon="🚗", layout="wide")
st.title("Carithm Visual Inspector")
st.caption("Upload a clear photo of visible vehicle damage for an initial repair estimate.")


@st.cache_resource(show_spinner="Loading the damage detector…")
def detector():
    return load_detector()


upload = st.file_uploader("Vehicle photo", type=["jpg", "jpeg", "png", "webp"])
if upload is not None:
    photo = Image.open(BytesIO(upload.getvalue())).convert("RGB")
    st.image(photo, caption="Uploaded photo", use_container_width=True)

    if st.button("Inspect damage", type="primary"):
        try:
            detections = detector().inspect(photo)
        except DetectorUnavailable as error:
            st.error(f"The damage model is unavailable: {error}")
            st.info("The Streamlit interface is ready. Connect a supported detector or hosted inference endpoint to enable inspection.")
            st.stop()

        if not detections:
            st.info("No supported damage type was detected. Try a closer, well-lit photo of the affected area.")
            st.stop()

        st.subheader("Inspection results")
        for detection in detections:
            repair = estimate(detection)
            with st.container(border=True):
                left, middle, right = st.columns(3)
                left.metric("Damage", detection.damage_type.replace("_", " ").title())
                middle.metric("Model confidence", f"{detection.confidence:.0%}")
                right.metric("Repair range", f"${repair.low_usd:,}–${repair.high_usd:,}")
                st.write(f"**Complexity:** {repair.complexity.title()}")
                st.write(f"**Driveable:** {'Yes' if repair.driveable else 'No — arrange an inspection or recovery.'}")
                st.write(f"**Common parts:** {', '.join(repair.parts)}")
                st.warning(repair.safety_note)

st.divider()
st.caption("Image-only estimate. It cannot find hidden, structural, or mechanical damage. Get an in-person inspection for safety decisions.")
