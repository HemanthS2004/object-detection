import gradio as gr
import cv2
import tempfile
from ultralytics import YOLO
from collections import Counter
import pandas as pd

def get_detection_stats(results):
    # Get detection statistics
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
    return stats_df, avg_confidence

def run_yolo(image=None, video=None, model_name="yolov8n", image_size=640, conf_threshold=0.25):
    # Load YOLO model
    model = YOLO(model_name)
    
    if image is not None:
        # Process image
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        annotated_image = results[0].plot()
        stats_df, avg_confidence = get_detection_stats(results)
        return (
            annotated_image[:, :, ::-1],  # Convert BGR to RGB for display
            None,  # video output
            stats_df,
            f"{avg_confidence:.2%}"
        )

    elif video is not None:
        # Process video
        video_path = tempfile.mktemp(suffix=".mp4")
        with open(video_path, "wb") as f:
            f.write(video.read())

        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = tempfile.mktemp(suffix=".mp4")
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        total_stats = Counter()
        total_confidence = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
            # Accumulate statistics
            frame_stats, frame_conf = get_detection_stats(results)
            for idx, row in frame_stats.iterrows():
                total_stats[row['Class']] += row['Count']
            total_confidence += frame_conf
            frame_count += 1

        cap.release()
        out.release()

        # Prepare final statistics
        stats_df = pd.DataFrame({
            'Class': list(total_stats.keys()),
            'Count': list(total_stats.values())
        })
        avg_confidence = (total_confidence / frame_count) if frame_count > 0 else 0

        return None, output_path, stats_df, f"{avg_confidence:.2%}"

    return None, None, pd.DataFrame(), "0%"

def run_realtime_yolo(model_name: str, image_size: int, conf_threshold: float):
    model_rt = YOLO(model_name)
    cap = cv2.VideoCapture(0)  # Open default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model_rt.predict(source=frame, imgsz=image_size, conf=conf_threshold)
        annotated_frame = results[0].plot()

        # Get live stats for current frame
        class_counts = Counter()
        for detection in results[0].boxes:
            class_name = results[0].names[int(detection.cls)]
            class_counts[class_name] += 1
        stats_df = pd.DataFrame({
            'Class': list(class_counts.keys()),
            'Count': list(class_counts.values())
        })
        avg_confidence = sum(float(detection.conf) for detection in results[0].boxes) / len(results[0].boxes) if results[0].boxes else 0

        # Return annotated frame and stats - adapted for Gradio streaming
        yield (
            annotated_frame[:, :, ::-1],  # Convert BGR to RGB
            None,
            stats_df,
            f"{avg_confidence:.2%}"
        )
        
    cap.release()

# Define Gradio interface
def create_app():
    with gr.Blocks() as app:
        gr.Markdown("### YOLO Object Detection")
        
        with gr.Row():
            with gr.Column():
                input_type = gr.Radio(["Image", "Video", "Real-time"], label="Select Input Type", value="Image")
                image_input = gr.Image(label="Upload Image", visible=True, type="pil")
                video_input = gr.Video(label="Upload Video", visible=False)
                model_name = gr.Dropdown(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                                         label="Model", value="yolov8n")
                image_size = gr.Slider(320, 1280, step=32, value=640, label="Image Size")
                conf_threshold = gr.Slider(0.0, 1.0, step=0.05, value=0.25, label="Confidence Threshold")
                detect_button = gr.Button("Run Detection")
            
            with gr.Column():
                output_image = gr.Image(label="Detected Image", visible=True)
                output_video = gr.Video(label="Detected Video", visible=False)
                stats_df = gr.Dataframe(label="Detection Statistics", headers=["Class", "Count"])
                avg_confidence = gr.Textbox(label="Average Confidence Score")
        
        # Change visibility based on input type
        def update_inputs(input_type):
            if input_type == "Image":
                return gr.update(visible=True), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            elif input_type == "Video":
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=True)
            else:  # Real-time
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
        
        input_type.change(fn=update_inputs, inputs=[input_type],
                          outputs=[image_input, video_input, output_image, output_video])
        
        # Run inference
        def detect(input_type, image, video, model_name, image_size, conf_threshold):
            if input_type == "Image" and image is not None:
                return run_yolo(image=image, model_name=model_name, image_size=image_size, conf_threshold=conf_threshold)
            elif input_type == "Video" and video is not None:
                return run_yolo(video=video, model_name=model_name, image_size=image_size, conf_threshold=conf_threshold)
            elif input_type == "Real-time":  # Real-time detection
                return run_realtime_yolo(model_name, image_size, conf_threshold)
            return None, None, pd.DataFrame(), "0%"
        
        detect_button.click(detect,
                            inputs=[input_type, image_input, video_input, model_name, image_size, conf_threshold],
                            outputs=[output_image, output_video, stats_df, avg_confidence])
    
    return app

# Launch the app
if __name__ == "__main__":
    def run_realtime_yolo_stream(model_name, image_size, conf_threshold):
        model_rt = YOLO(model_name)
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model_rt.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            annotated_frame = results[0].plot()
            class_counts = Counter()
            for detection in results[0].boxes:
                class_name = results[0].names[int(detection.cls)]
                class_counts[class_name] += 1
            stats_df = pd.DataFrame({
                'Class': list(class_counts.keys()),
                'Count': list(class_counts.values())
            })
            avg_confidence = sum(float(detection.conf) for detection in results[0].boxes) / len(results[0].boxes) if results[0].boxes else 0
            yield annotated_frame[:, :, ::-1], None, stats_df, f"{avg_confidence:.2%}"
        cap.release()

    app = create_app()
    
    # Override detect button for real-time streaming
    def detect_stream(input_type, image, video, model_name, image_size, conf_threshold):
        if input_type == "Image" and image is not None:
            return run_yolo(image=image, model_name=model_name, image_size=image_size, conf_threshold=conf_threshold)
        elif input_type == "Video" and video is not None:
            return run_yolo(video=video, model_name=model_name, image_size=image_size, conf_threshold=conf_threshold)
        elif input_type == "Real-time":
            return run_realtime_yolo_stream(model_name, image_size, conf_threshold)
        return None, None, pd.DataFrame(), "0%"
    
    app.load(detect_stream)
    app.launch()
