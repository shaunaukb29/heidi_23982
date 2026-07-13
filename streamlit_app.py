from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image

from carithm_inspector.detector import DetectorUnavailable, load_detector
from carithm_inspector.domain import Detection, Estimate, Priority, Severity, ViewAngle
from carithm_inspector.estimates import estimate
from carithm_inspector.visualization import (
    PRIORITY_LABELS,
    SEVERITY_ICONS,
    format_damage_type,
    render_inspection_overlay,
)

st.set_page_config(page_title="Carithm Inspector", page_icon="🚗", layout="wide")

st.markdown(
    """
    <style>
    .damage-card { padding: 0.5rem 0; border-radius: 8px; }
    .damage-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.25rem; color: #4da6ff; }
    .muted { color: #666; font-size: 0.9rem; }
    .stRadio [role=radiogroup] { padding: 10px; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Carithm Visual Inspector")
st.caption("Upload a clear photo and specify the camera angle for a professional inspection report.")


@st.cache_resource(show_spinner="Loading the damage detector…")
def detector():
    return load_detector()


def _severity_label(severity: Severity) -> str:
    return f"{SEVERITY_ICONS[severity]} {severity.value.title()}"


def _priority_block(priority: Priority) -> None:
    label, detail = PRIORITY_LABELS[priority]
    st.markdown(f"**{label}**")
    st.caption(detail)


def _render_damage_card(
    index: int,
    detection: Detection,
    repair: Estimate,
    *,
    selected: bool,
) -> None:
    # Use the dynamic currency from the domain model
    currency = repair.cost_breakdown.currency

    with st.container(border=True):
        st.markdown(
            f'<div class="damage-title">{"✓ " if selected else ""}'
            f'{format_damage_type(detection.damage_type)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Confidence:** {detection.confidence:.0%}")
        st.markdown(f"**Severity:** {_severity_label(repair.severity)}")
        st.markdown(f"**Surface area:** {detection.area_ratio:.1%}")

        st.markdown("**Location**")
        st.write(repair.vehicle_part.location)
        st.markdown("**Panel**")
        st.caption(f"{repair.vehicle_part.panel} · {repair.vehicle_part.material}")

        st.markdown("**Recommended repair**")
        for step in repair.repair_steps:
            st.markdown(f"✔ {step}")

        st.markdown("**Estimated shop time**")
        st.write(f"{repair.shop_time_low_hours:.1f} – {repair.shop_time_high_hours:.1f} hours")

        st.markdown("**Priority**")
        _priority_block(repair.priority)

        st.markdown("**Cost breakdown**")
        breakdown = repair.cost_breakdown
        col1, col2 = st.columns(2)
        
        # Updated to use the new domain.py attributes and dynamic currency
        col1.write(f"Labour: **{breakdown.labour_cost:,.0f} {currency}**")
        col1.write(f"Paint: **{breakdown.paint_cost:,.0f} {currency}**")
        col2.write(f"Parts: **{breakdown.parts_cost:,.0f} {currency}**")
        col2.write(f"**Total: {breakdown.total_cost:,.0f} {currency}**")
        
        st.caption(f"Range: {repair.low_cost:,.0f}–{repair.high_cost:,.0f} {currency}")

        if not repair.driveable:
            st.error(repair.safety_note)
        elif repair.priority == Priority.SAFETY_CRITICAL:
            st.warning(repair.safety_note)
        else:
            st.info(repair.safety_note)


upload = st.file_uploader("Vehicle photo", type=["jpg", "jpeg", "png", "webp"])

if upload is not None:
    photo = Image.open(BytesIO(upload.getvalue())).convert("RGB")

    # Inject the crucial ViewAngle selector before running inference
    st.markdown("### Camera Perspective")
    selected_angle_str = st.radio(
        "Which side of the vehicle is shown in this photo?",
        options=[v.value for v in ViewAngle],
        horizontal=True,
        index=4 # Defaults to Unknown
    )
    
    if "inspection" not in st.session_state or st.session_state.get("upload_name") != upload.name:
        st.session_state.upload_name = upload.name
        st.session_state.inspection = None
        st.session_state.selected_damage = 0

    if st.button("Run professional inspection", type="primary"):
        try:
            view_angle = ViewAngle(selected_angle_str)
            
            # The ML model only takes the photo to find the bounding boxes
            detections = detector().inspect(photo)
            
            # The heuristic engine takes the angle to map the location
            reports = [(detection, estimate(detection, view_angle)) for detection in detections]
            
            st.session_state.inspection = reports
            st.session_state.selected_damage = 0
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

    reports: list[tuple[Detection, Estimate]] = st.session_state.inspection

    if not reports:
        st.image(photo, caption="Uploaded photo", use_container_width=True)
        st.info("No supported damage type was detected. Try a closer, well-lit photo of the affected area.")
        st.stop()

    detections = [item[0] for item in reports]
    selected = st.session_state.selected_damage
    if selected >= len(reports):
        selected = 0
        st.session_state.selected_damage = 0

    overlay = render_inspection_overlay(photo, detections, selected_index=selected)

    st.subheader("Professional inspection")
    st.caption(f"{len(reports)} damage{'s' if len(reports) != 1 else ''} detected")

    image_col, list_col = st.columns([3, 2], gap="large")

    with image_col:
        st.image(overlay, caption="Damage overlay — select a finding to highlight", use_container_width=True)
        st.caption("Bounding boxes and segmentation masks show detected damage areas.")

    with list_col:
        st.markdown("**Detected damages**")
        for index, (detection, repair) in enumerate(reports):
            label = (
                f"{'✓ ' if index == selected else ''}"
                f"{format_damage_type(detection.damage_type)} "
                f"({detection.confidence:.0%}) — {repair.vehicle_part.location}"
            )
            if st.button(label, key=f"damage_select_{index}", use_container_width=True):
                st.session_state.selected_damage = index
                st.rerun()

        st.divider()
        _render_damage_card(
            selected,
            reports[selected][0],
            reports[selected][1],
            selected=True,
        )

    if len(reports) > 1:
        st.divider()
        st.markdown("**All findings**")
        cols = st.columns(min(len(reports), 3))
        for index, (detection, repair) in enumerate(reports):
            with cols[index % len(cols)]:
                _render_damage_card(index, detection, repair, selected=index == selected)

st.divider()
st.caption(
    "Image-only estimate. It cannot find hidden, structural, or mechanical damage. "
    "Get an in-person inspection for safety decisions."
)
