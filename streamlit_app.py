from pathlib import Path

import streamlit as st
from PIL import Image

from main import (
    MATCH_THRESHOLD,
    add_criminal_record,
    delete_criminal_record,
    detect_criminal_record,
    image_bytes_to_bgr,
    list_criminals,
    list_detections,
    parse_optional_age,
)


st.set_page_config(
    page_title="Criminal Face Detection System",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles():
    st.markdown(
        """
        <style>
        :root {
            --app-text: #1f2937;
            --muted-text: #475569;
            --panel-bg: rgba(255, 248, 240, 0.92);
            --panel-border: rgba(217, 119, 6, 0.18);
            --input-bg: rgba(255, 252, 248, 0.98);
            --accent: #0f766e;
            --accent-deep: #115e59;
            --accent-warm: #c2410c;
            --hero-start: rgba(28, 25, 23, 0.97);
            --hero-end: rgba(120, 53, 15, 0.92);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(245, 158, 11, 0.2), transparent 26%),
                radial-gradient(circle at top right, rgba(20, 184, 166, 0.18), transparent 24%),
                linear-gradient(180deg, #fff7ed 0%, #f8fafc 52%, #ecfeff 100%);
            color: var(--app-text);
        }
        .stApp, .stApp p, .stApp label, .stApp span, .stApp div, .stApp li {
            color: var(--app-text);
        }
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.78);
            border-right: 1px solid var(--panel-border);
        }
        [data-testid="stSidebar"] * {
            color: var(--app-text);
        }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"], .stCaption, small {
            color: var(--muted-text) !important;
        }
        [data-testid="stFileUploader"] label,
        [data-testid="stCameraInput"] label,
        [data-testid="stTextInput"] label,
        .stSelectbox label,
        .stRadio label,
        .stMarkdown,
        .stSubheader,
        h1, h2, h3 {
            color: var(--app-text) !important;
        }
        .stTextInput input,
        .stTextArea textarea,
        .stNumberInput input,
        [data-testid="stFileUploaderDropzone"],
        [data-baseweb="select"] > div {
            background: var(--input-bg) !important;
            color: var(--app-text) !important;
            border-color: var(--panel-border) !important;
        }
        .stTextInput input::placeholder,
        .stTextArea textarea::placeholder {
            color: #64748b !important;
        }
        [data-testid="stFileUploaderDropzone"] * {
            color: var(--muted-text) !important;
        }
        .stButton > button,
        .stFormSubmitButton > button {
            background: linear-gradient(135deg, var(--accent), var(--accent-deep)) !important;
            color: #f8fafc !important;
            border: 1px solid rgba(13, 148, 136, 0.28) !important;
            border-radius: 12px !important;
            box-shadow: 0 12px 24px rgba(17, 94, 89, 0.16);
            transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
        }
        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            filter: brightness(1.05);
            transform: translateY(-1px);
            box-shadow: 0 16px 30px rgba(17, 94, 89, 0.2);
        }
        .stButton > button:focus,
        .stFormSubmitButton > button:focus {
            color: #f8fafc !important;
            border-color: rgba(20, 184, 166, 0.55) !important;
            box-shadow: 0 0 0 0.2rem rgba(45, 212, 191, 0.2) !important;
        }
        .stButton > button[kind="secondary"] {
            background: linear-gradient(135deg, var(--accent-warm), #9a3412) !important;
            border-color: rgba(194, 65, 12, 0.35) !important;
        }
        .stButton > button:disabled,
        .stFormSubmitButton > button:disabled {
            background: #94a3b8 !important;
            color: #e2e8f0 !important;
            border-color: #94a3b8 !important;
            box-shadow: none !important;
            transform: none !important;
        }
        [data-testid="stVerticalBlock"] [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel-bg);
            border-radius: 16px;
            border: 1px solid var(--panel-border);
        }
        .hero {
            padding: 1.6rem 1.8rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgb(41 36 34 / 97%), rgb(255, 108, 108));
            color: white;
            border: 1px solid rgba(251, 191, 36, 0.16);
            box-shadow: 0 24px 60px rgba(120, 53, 15, 0.18);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
        }
        .hero p {
            margin: 0.6rem 0 0 0;
            color: #cbd5e1;
            font-size: 1rem;
        }
        .panel {
            background: var(--panel-bg);
            border: 1px solid var(--panel-border);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 14px 36px rgba(15, 23, 42, 0.08);
        }
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
            color: var(--app-text);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_header():
    st.markdown(
        """
        <div class="hero">
            <h1>Criminal Face Detection System</h1>
            <p>Streamlit deployment with face detection, criminal registry, and detection history in one Python app.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_from_upload(uploaded_file):
    if not uploaded_file:
        return None, None
    image_bytes = uploaded_file.getvalue()
    preview = Image.open(uploaded_file).convert("RGB")
    uploaded_file.seek(0)
    image = image_bytes_to_bgr(image_bytes)
    return preview, image


def show_result_card(result):
    if not result:
        return
    if result.get("error"):
        st.error(result["error"])
        return
    if result.get("match"):
        criminal = result["criminal"]
        st.error(
            f"Match found: {criminal['name']} | Crime: {criminal['crime']} | "
            f"Confidence: {criminal['confidence']:.2f}%"
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("Age", criminal.get("age") or "N/A")
        col2.metric("Location", criminal.get("location") or "Unknown")
        col3.metric("Confidence", f"{criminal['confidence']:.2f}%")
    else:
        st.success(result.get("message", "No match found"))


def render_detect_section():
    st.markdown('<div class="section-title">Detect From Image</div>', unsafe_allow_html=True)
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        uploaded = st.file_uploader(
            "Upload a face image",
            type=["png", "jpg", "jpeg"],
            key="detect_upload",
        )
        camera_image = st.camera_input("Or capture from camera", key="detect_camera")
        active_input = camera_image or uploaded
        preview, image = image_from_upload(active_input)
        if preview:
            st.image(preview, use_container_width=True)

    with right:
        detection_location = st.text_input(
            "Detection location",
            value="Streamlit Application",
            key="detect_location",
        )
        if st.button("Run Detection", type="primary", use_container_width=True):
            if image is None:
                st.warning("Upload or capture an image first.")
            else:
                with st.spinner("Analyzing face..."):
                    result = detect_criminal_record(image, detection_location.strip() or "Unknown")
                st.session_state["detect_result"] = result

        show_result_card(st.session_state.get("detect_result"))


def render_add_section():
    st.markdown('<div class="section-title">Add Criminal To Database</div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="large")

    with left:
        uploaded = st.file_uploader(
            "Upload criminal photo",
            type=["png", "jpg", "jpeg"],
            key="add_upload",
        )
        preview, image = image_from_upload(uploaded)
        if preview:
            st.image(preview, use_container_width=True)

    with right:
        with st.form("add_criminal_form", clear_on_submit=True):
            name = st.text_input("Full name")
            crime = st.text_input("Crime")
            age_raw = st.text_input("Age")
            location = st.text_input("Last known location")
            submitted = st.form_submit_button("Add Criminal", type="primary", use_container_width=True)

        if submitted:
            if image is None:
                st.warning("Upload a criminal photo first.")
                return
            if not name.strip() or not crime.strip():
                st.warning("Name and crime are required.")
                return
            try:
                age = parse_optional_age(age_raw.strip())
                with st.spinner("Saving record..."):
                    result = add_criminal_record(
                        name=name.strip(),
                        crime=crime.strip(),
                        age=age,
                        location=location.strip() or None,
                        image=image,
                    )
            except ValueError as exc:
                st.error(str(exc))
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                st.success(f"{result['name']} added successfully.")
                st.session_state.pop("add_upload", None)


def render_database_section():
    criminals = list_criminals()
    st.markdown('<div class="section-title">Criminal Database</div>', unsafe_allow_html=True)
    st.caption(f"{len(criminals)} records")

    if not criminals:
        st.info("No criminals in the database yet.")
        return

    for criminal in criminals:
        with st.container(border=True):
            cols = st.columns([1.1, 1.4, 0.7], gap="large")
            with cols[0]:
                image_path = criminal.get("image_path")
                if image_path and Path(image_path).exists():
                    st.image(str(image_path), use_container_width=True)
            with cols[1]:
                st.subheader(criminal["name"])
                st.write(f"Crime: {criminal['crime']}")
                st.write(f"Age: {criminal.get('age') or 'N/A'}")
                st.write(f"Location: {criminal.get('location') or 'Unknown'}")
                st.caption(f"Added: {criminal['date_added']}")
            with cols[2]:
                if st.button("Delete", key=f"delete_{criminal['id']}", use_container_width=True):
                    try:
                        delete_criminal_record(criminal["id"])
                    except LookupError as exc:
                        st.error(str(exc))
                    else:
                        st.success(f"Deleted {criminal['name']}.")
                        st.rerun()


def render_history_section():
    detections = list_detections()
    st.markdown('<div class="section-title">Detection History</div>', unsafe_allow_html=True)
    st.caption(f"{len(detections)} recent detections")

    if not detections:
        st.info("No detection history yet.")
        return

    for detection in detections:
        with st.container(border=True):
            left, right = st.columns([1.6, 0.8])
            left.write(f"**{detection['criminal_name']}**")
            left.write(f"Crime: {detection['crime']}")
            left.write(f"Location: {detection.get('location') or 'Unknown'}")
            right.metric("Confidence", f"{detection['confidence']:.2f}%")
            right.caption(detection["detection_time"])


def render_sidebar():
    st.sidebar.title("Navigation")
    return st.sidebar.radio(
        "Choose a section",
        ["Detect", "Add Criminal", "Database", "History"],
        label_visibility="collapsed",
    )


def main():
    inject_styles()
    show_header()

    criminals = list_criminals()
    detections = list_detections()
    top1, top2, top3 = st.columns(3)
    top1.metric("Database Records", len(criminals))
    top2.metric("Detection Logs", len(detections))
    top3.metric("Match Threshold", f"{MATCH_THRESHOLD * 100:.0f}%")

    section = render_sidebar()
    if section == "Detect":
        render_detect_section()
    elif section == "Add Criminal":
        render_add_section()
    elif section == "Database":
        render_database_section()
    else:
        render_history_section()


if __name__ == "__main__":
    main()
