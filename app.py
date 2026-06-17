import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from collections import Counter
import pandas as pd
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="YOLOv8 Object Detector",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern minimal design
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
    }
    
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        min-height: 100vh;
    }
    
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
    }
    
    .header-title {
        font-size: 42px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    
    .subheader-text {
        color: #666;
        font-size: 16px;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0

def get_detection_stats(results):
    """Extract detection statistics from YOLO results"""
    class_counts = Counter()
    confidences = []
    
    for detection in results[0].boxes:
        class_name = results[0].names[int(detection.cls)]
        class_counts[class_name] += 1
        confidences.append(float(detection.conf))
    
    stats_df = pd.DataFrame({
        'Class': list(class_counts.keys()),
        'Count': list(class_counts.values())
    })
    
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    return stats_df, avg_confidence, len(results[0].boxes)

def process_image(image, model_name, image_size, conf_threshold):
    """Process image with YOLO"""
    model = YOLO(model_name)
    results = model.predict(source=image, imgsz=image_size, conf=conf_threshold, verbose=False)
    annotated_image = results[0].plot()
    stats_df, avg_confidence, detection_count = get_detection_stats(results)
    return annotated_image[:, :, ::-1], stats_df, avg_confidence, detection_count

def process_video(video_path, model_name, image_size, conf_threshold, progress_bar):
    """Process video with YOLO"""
    model = YOLO(model_name)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path = tempfile.mktemp(suffix=".mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    total_stats = Counter()
    total_confidence = 0
    frame_count = 0
    total_detections = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        
        frame_stats, frame_conf, detections = get_detection_stats(results)
        for idx, row in frame_stats.iterrows():
            total_stats[row['Class']] += row['Count']
        total_confidence += frame_conf
        total_detections += detections
        frame_count += 1
        
        progress_bar.progress(min(frame_count / total_frames, 1.0))
    
    cap.release()
    out.release()
    
    stats_df = pd.DataFrame({
        'Class': list(total_stats.keys()),
        'Count': list(total_stats.values())
    })
    avg_confidence = (total_confidence / frame_count) if frame_count > 0 else 0
    
    return output_path, stats_df, avg_confidence, total_detections

# Main UI
st.markdown('<div class="header-title">🎯 YOLOv8 Object Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader-text">Real-time object detection with advanced analytics</div>', unsafe_allow_html=True)

# Tab interface
tab1, tab2, tab3 = st.tabs(["📷 Image", "🎥 Video", "⚙️ Settings"])

with tab3:
    st.subheader("Detection Settings")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        model_name = st.selectbox(
            "YOLO Model",
            ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
            help="Larger models are more accurate but slower"
        )
    
    with col2:
        image_size = st.slider(
            "Image Size",
            min_value=320,
            max_value=1280,
            value=640,
            step=32,
            help="Higher sizes may improve accuracy but increase processing time"
        )
    
    with col3:
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Lower values detect more objects (higher false positives)"
        )

with tab1:
    st.subheader("Image Detection")
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### Upload Image")
        uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])
        
        if uploaded_image:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔍 Detect Objects", key="img_detect"):
                with st.spinner("Processing image..."):
                    processed_img, stats, confidence, count = process_image(
                        image, model_name, image_size, conf_threshold
                    )
                    st.session_state.processed_image = processed_img
                    st.session_state.stats = stats
                    st.session_state.confidence = confidence
    
    with col2:
        st.markdown("### Detection Results")
        if st.session_state.processed_image is not None:
            st.image(st.session_state.processed_image, caption="Detected Objects", use_column_width=True)
            
            # Metrics
            result_col1, result_col2, result_col3 = st.columns(3)
            with result_col1:
                st.metric("Detections", len(st.session_state.stats))
            with result_col2:
                st.metric("Confidence", f"{st.session_state.confidence:.1%}")
            with result_col3:
                st.metric("Status", "✅ Complete")
            
            # Statistics table
            st.markdown("### Detection Summary")
            st.dataframe(st.session_state.stats, use_container_width=True, hide_index=True)
        else:
            st.info("👆 Upload an image and click 'Detect Objects' to see results")

with tab2:
    st.subheader("Video Detection")
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### Upload Video")
        uploaded_video = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_video:
            st.video(uploaded_video)
            
            if st.button("🎬 Process Video", key="vid_detect"):
                with st.spinner("Processing video..."):
                    video_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    video_temp.write(uploaded_video.read())
                    video_temp.close()
                    
                    progress_bar = st.progress(0)
                    output_path, stats, confidence, detections = process_video(
                        video_temp.name, model_name, image_size, conf_threshold, progress_bar
                    )
                    
                    st.session_state.stats = stats
                    st.session_state.confidence = confidence
                    st.session_state.processed_image = output_path
                    
                    os.unlink(video_temp.name)
    
    with col2:
        st.markdown("### Processing Results")
        if st.session_state.processed_image and isinstance(st.session_state.processed_image, str):
            st.video(st.session_state.processed_image)
            
            # Metrics
            result_col1, result_col2, result_col3 = st.columns(3)
            with result_col1:
                st.metric("Total Detections", len(st.session_state.stats) if st.session_state.stats is not None else 0)
            with result_col2:
                st.metric("Avg Confidence", f"{st.session_state.confidence:.1%}")
            with result_col3:
                st.metric("Status", "✅ Complete")
            
            # Statistics table
            st.markdown("### Detection Summary")
            st.dataframe(st.session_state.stats, use_container_width=True, hide_index=True)
        else:
            st.info("👆 Upload a video and click 'Process Video' to see results")
            return run_yolo(video=video, model_name=model_name, image_size=image_size, conf_threshold=conf_threshold)
        elif input_type == "Real-time":
            return run_realtime_yolo_stream(model_name, image_size, conf_threshold)
        return None, None, pd.DataFrame(), "0%"
    
    app.load(detect_stream)
    app.launch()
