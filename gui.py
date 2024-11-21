import streamlit as st
from ultralytics import YOLO
import os
import cv2

# Function to perform inference on videos
def run_video_inference(model_path, video_path, output_dir="output"):
    try:
        # Load YOLO model
        model = YOLO(model_path)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Unable to open video file.")
            return None

        # Output settings
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "output_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # Codec for MP4 (browser-friendly)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process video frame by frame
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        for frame_idx in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLO inference on the frame
            results = model.predict(frame, conf=0.25, device="cpu")
            annotated_frame = results[0].plot()  # Annotated frame with detections
            out.write(annotated_frame)  # Save the frame to the output video

            # Update progress bar
            progress_bar.progress((frame_idx + 1) / frame_count)

        # Release resources
        cap.release()
        out.release()
        progress_bar.empty()
        return output_path
    except Exception as e:
        st.error(f"Error during video inference: {e}")
        return None

# Path to YOLO model
MODEL_PATH = "C:/Users/DELL/Downloads/yolov8-VER-3/best_p.pt"

# Ensure 'temp' directory exists
TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Streamlit app
st.title("YOLOv8 Image/Video Inference")
st.sidebar.header("Upload Input")
st.sidebar.write("Upload an image or video to test.")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Choose an image or video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

if uploaded_file is not None:
    # Display uploaded file
    st.sidebar.write("Uploaded File:")
    file_name = uploaded_file.name
    st.sidebar.write(file_name)

    # Save the file temporarily
    temp_file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Check if the file is an image or video
    if file_name.lower().endswith(("jpg", "jpeg", "png")):
        # Display image on the left
        st.image(temp_file_path, caption="Uploaded Image", use_column_width=True)

        # Run inference
        st.write("Running inference...")
        results = YOLO(MODEL_PATH).predict(temp_file_path, conf=0.25, device="cpu")

        if results:
            # Convert BGR to RGB for proper display
            result_image = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

            # Display results on the right
            st.write("Inference Results:")
            st.image(result_image, caption="Detected Objects", use_column_width=True)

    elif file_name.lower().endswith(("mp4", "avi", "mov")):
        # Display video on the left
        st.video(temp_file_path)

        # Run inference
        st.write("Running video inference... This may take some time.")
        output_video_path = run_video_inference(MODEL_PATH, temp_file_path)

        if output_video_path and os.path.exists(output_video_path):
            st.write("Inference completed. Processed video:")
            # Display processed video
            with open(output_video_path, "rb") as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
        else:
            st.error("Error: Unable to process the video.")
    else:
        st.error("Unsupported file format! Please upload a valid image or video file.")

else:
    st.write("Awaiting file upload...")
