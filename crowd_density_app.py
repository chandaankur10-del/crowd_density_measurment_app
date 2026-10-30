import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
import subprocess

# Set up Page configuration
st.set_page_config(page_title="Crowd Density Estimator", page_icon="👥", layout="centered")

st.title("👥 AI Crowd Density Estimator")
st.write("Upload a crowd video, and this application will count people and compute crowd density in real-time using YOLO!")

# Sidebar options
st.sidebar.header("Configuration")
physical_area = st.sidebar.slider("Estimated Physical Area (sq. meters)", min_value=1.0, max_value=500.0, value=25.0, step=1.0)
confidence_threshold = st.sidebar.slider("YOLO Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
max_frames = st.sidebar.number_input("Max Frames to Process (for demo speed)", min_value=10, max_value=1000, value=150, step=10)

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO('yolo11n.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")

uploaded_file = st.file_uploader("Choose a video file (MP4 format preferred)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()

    st.video(tfile.name)
    
    if st.button("🚀 Process Video"):
        cap = cv2.VideoCapture(tfile.name)
        
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
        else:
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            # Setup output temporary file path
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            frame_count = 0
            total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)
            
            while cap.isOpened() and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run detection
                results = model(frame, conf=confidence_threshold, verbose=False)
                
                person_count = 0
                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls[0])
                        if cls == 0:  # Class 'person'
                            person_count += 1
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Calculate Density
                density = person_count / physical_area
                
                # Annotation overlay
                cv2.putText(frame, f"Count: {person_count}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Density: {density:.3f} p/m2", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                out.write(frame)
                frame_count += 1
                
                # Update progress
                progress_bar.progress(frame_count / total_frames)
                status_text.text(f"Processing frame {frame_count}/{total_frames}... (Current Count: {person_count})")
            
            cap.release()
            out.release()
            status_text.text("Conversion completed! Finalizing video stream...")
            
            # Re-encode to H.264 using ffmpeg so it displays correctly inside the Streamlit web browser
            h264_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            ffmpeg_cmd = f"ffmpeg -y -i {temp_output} -vcodec libx264 -pix_fmt yuv420p {h264_output}"
            subprocess.run(ffmpeg_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            st.success("🎉 Video successfully processed!")
            
            # Display annotated output video
            with open(h264_output, 'rb') as f:
                video_bytes = f.read()
            st.video(video_bytes)
            
            # Provide a download button for the processed file
            st.download_button(
                label="📥 Download Annotated Video",
                data=video_bytes,
                file_name="annotated_crowd_density.mp4",
                mime="video/mp4"
            )
            
            # Clean up files
            os.remove(tfile.name)
            os.remove(temp_output)
            os.remove(h264_output)
