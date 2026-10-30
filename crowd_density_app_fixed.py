import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="Crowd Density Estimator",
    page_icon="👥",
    layout="centered",
)

st.title("👥 AI Crowd Density Estimator")
st.write(
    "Upload a video to detect people, count them frame by frame, "
    "and estimate crowd density."
)

# Put crowd_density_best.pt in the same GitHub repository folder as this app.
MODEL_PATH = Path(__file__).parent / "crowd_density_best.pt"


@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


with st.sidebar:
    st.header("Configuration")
    physical_area = st.number_input(
        "Estimated physical area (m²)",
        min_value=1.0,
        max_value=100000.0,
        value=25.0,
        step=1.0,
    )
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.05,
    )
    max_frames = st.number_input(
        "Maximum frames to process",
        min_value=10,
        max_value=5000,
        value=150,
        step=10,
    )

if not MODEL_PATH.is_file():
    st.error(
        f"Model file not found: {MODEL_PATH.name}. "
        "Upload crowd_density_best.pt to the same GitHub repository "
        "folder as crowd_density_app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the YOLO model: {exc}")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_file is not None:
    input_path = None
    output_path = None

    try:
        # Keep the uploaded video's original extension.
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_in:
            temp_in.write(uploaded_file.getbuffer())
            input_path = temp_in.name

        st.video(input_path)

        if st.button("🚀 Process video", type="primary"):
            cap = cv2.VideoCapture(input_path)

            if not cap.isOpened():
                st.error("OpenCV could not open this video.")
                cap.release()
                st.stop()

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                fps = 25.0

            source_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames = (
                min(source_frame_count, int(max_frames))
                if source_frame_count > 0
                else int(max_frames)
            )

            if width <= 0 or height <= 0 or total_frames <= 0:
                cap.release()
                st.error("The uploaded video has invalid dimensions or no frames.")
                st.stop()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_out:
                output_path = temp_out.name

            # mp4v is broadly available in OpenCV builds.
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )

            if not writer.isOpened():
                cap.release()
                st.error("Could not create the output video.")
                st.stop()

            progress = st.progress(0)
            status = st.empty()
            frame_count = 0
            latest_count = 0

            try:
                while frame_count < total_frames:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.predict(
                        frame,
                        conf=confidence_threshold,
                        verbose=False,
                    )

                    person_count = 0
                    for result in results:
                        if result.boxes is None:
                            continue

                        for box in result.boxes:
                            class_id = int(box.cls.item())
                            class_name = str(result.names.get(class_id, class_id)).lower()

                            # This app counts the "person" class only.
                            if class_name == "person":
                                person_count += 1
                                x1, y1, x2, y2 = map(
                                    int, box.xyxy[0].cpu().tolist()
                                )
                                cv2.rectangle(
                                    frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                                )

                    density = person_count / float(physical_area)
                    cv2.putText(
                        frame,
                        f"People: {person_count}",
                        (25, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Density: {density:.3f} people/m2",
                        (25, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                    writer.write(frame)
                    frame_count += 1
                    latest_count = person_count
                    progress.progress(min(frame_count / total_frames, 1.0))
                    status.text(
                        f"Processed {frame_count}/{total_frames} frames | "
                        f"Current people count: {latest_count}"
                    )
            finally:
                cap.release()
                writer.release()

            if frame_count == 0 or not os.path.exists(output_path):
                st.error("No video frames were processed.")
                st.stop()

            st.success(
                f"Processing complete. Processed {frame_count} frames. "
                f"Last-frame count: {latest_count}."
            )

            with open(output_path, "rb") as video_file:
                video_bytes = video_file.read()

            st.video(video_bytes)
            st.download_button(
                "📥 Download annotated video",
                data=video_bytes,
                file_name="annotated_crowd_density.mp4",
                mime="video/mp4",
            )

    except Exception as exc:
        st.error(f"Video processing failed: {exc}")

    finally:
        for temporary_path in (input_path, output_path):
            if temporary_path and os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass
