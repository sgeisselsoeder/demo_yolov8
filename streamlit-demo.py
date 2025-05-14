import os
import gc
import cv2
import time
import torch
import numpy as np
import streamlit as st
from ultralytics import YOLO
from ultralytics import solutions

# Configuration Constants
APP_TITLE = "YOLO Object Detection Demo"
APP_DESCRIPTION = "This app uses YOLO to detect objects in real-time from your webcam."

# Page Configuration
PAGE_CONFIG = {
    "page_title": "YOLO Object Detection",
    "page_icon": "🔍",
    "layout": "wide",
}

# Stream Settings
DEFAULT_FPS_LIMIT = 15
DEFAULT_FRAME_SKIP = 2
MAX_CAMERAS = 10

# Model Settings
DEFAULT_CONFIDENCE = 0.25
DEFAULT_POSE_CONFIDENCE = 0.5
DEFAULT_SEG_CONFIDENCE = 0.3

# Resolution Settings
RESOLUTIONS = {
    "640x480 (VGA)": (640, 480),
    "1280x720 (HD)": (1280, 720),
    "1920x1080 (Full HD)": (1920, 1080),
    "2560x1440 (2K)": (2560, 1440),
    "3840x2160 (4K)": (3840, 2160),
}

# COCO Class names
COCO_CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


# Helper function to list available cameras
def list_available_cameras(max_cameras=MAX_CAMERAS):
    """Check for available camera devices up to max_cameras."""
    available_cameras = {}
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get camera name if possible
            ret, frame = cap.read()
            if ret:
                available_cameras[i] = f"Camera {i}"
            else:
                available_cameras[i] = f"Camera {i}"
            cap.release()
    return available_cameras


# Set page configuration
st.set_page_config(**PAGE_CONFIG)

# App title and description
st.title(APP_TITLE)
st.write(APP_DESCRIPTION)

# Get list of model files in the same directory
model_files = [f for f in os.listdir(".") if f.endswith(".pt")]

# Categorize models based on their capabilities
pose_models = [f for f in model_files if "pose" in f.lower()]
segment_models = [f for f in model_files if "seg" in f.lower()]
standard_models = [
    f for f in model_files if "pose" not in f.lower() and "seg" not in f.lower()
]

# Detection mode selection
detection_mode = st.sidebar.selectbox(
    "Detection Mode", ["Standard", "Heatmap", "Pose", "Segmentation"], index=0
)

# Show only relevant models based on the selected mode
if detection_mode == "Pose":
    if pose_models:
        model_options = pose_models
        model_version = st.sidebar.selectbox("Select Pose Model", model_options)
    else:
        st.sidebar.warning(
            "No pose-specific models found. Using standard models instead."
        )
        model_options = standard_models
        model_version = st.sidebar.selectbox("Select Model", model_options)
        st.sidebar.info("Download a pose model like yolo11n-pose.pt for best results.")
elif detection_mode == "Segmentation":
    if segment_models:
        model_options = segment_models
        model_version = st.sidebar.selectbox("Select Segmentation Model", model_options)
    else:
        st.sidebar.warning(
            "No segmentation-specific models found. Using standard models instead."
        )
        model_options = standard_models
        model_version = st.sidebar.selectbox("Select Model", model_options)
        st.sidebar.info(
            "Download a segmentation model like yolo11n-seg.pt for best results."
        )
else:  # Standard or Heatmap
    model_options = standard_models
    model_version = st.sidebar.selectbox("Select Detection Model", model_options)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_CONFIDENCE,
    step=0.05,
)

# FPS control
fps_limit = st.sidebar.slider(
    "FPS Limit", min_value=1, max_value=30, value=DEFAULT_FPS_LIMIT, step=1
)

# Warning if pose mode is selected but no pose model is selected
if detection_mode == "Pose" and "pose" not in model_version.lower():
    st.sidebar.warning(
        f"Selected model '{model_version}' may not be a pose model. For best results, use a model with 'pose' in the name."
    )

# Heatmap options (only shown when heatmap mode is selected)
if detection_mode == "Heatmap":
    heatmap_colormap = st.sidebar.selectbox(
        "Heatmap Colormap",
        ["COLORMAP_JET", "COLORMAP_TURBO", "COLORMAP_HOT", "COLORMAP_INFERNO"],
        index=0,
    )

    # Class selection for heatmap
    class_options = st.sidebar.multiselect(
        "Track Classes (empty = all)",
        options=COCO_CLASS_NAMES,
        default=["person"],
    )

    # Reset heatmap button
    reset_heatmap = st.sidebar.button("Reset Heatmap")

# Pose detection options (only shown when pose mode is selected)
if detection_mode == "Pose":
    pose_confidence = st.sidebar.slider(
        "Pose Confidence",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_POSE_CONFIDENCE,
        step=0.05,
    )

    show_keypoints = st.sidebar.checkbox("Show Keypoints", value=True)
    show_skeleton = st.sidebar.checkbox("Show Skeleton", value=True)

# Segmentation options (only shown when segmentation mode is selected)
if detection_mode == "Segmentation":
    # Segmentation confidence threshold
    seg_confidence = st.sidebar.slider(
        "Segmentation Confidence",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_SEG_CONFIDENCE,
        step=0.05,
    )

    # Segmentation visualization options
    show_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
    show_masks = st.sidebar.checkbox("Show Masks", value=True)

# Get available cameras
available_cameras = list_available_cameras()
if not available_cameras:
    st.error("No cameras detected. Please connect a camera and restart the app.")
    st.stop()

# Camera selection in sidebar
camera_id = st.sidebar.selectbox(
    "Select Camera",
    options=list(available_cameras.keys()),
    format_func=lambda x: available_cameras[x],
    index=0,
)

# Add resolution selector
selected_resolution = st.sidebar.selectbox(
    "Select Resolution", options=list(RESOLUTIONS.keys()), index=1  # Default to HD
)


# Load YOLO model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


def clear_memory():
    """Clear GPU and CPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Move model loading inside stream_active block
model = None

# Create two columns
col1, col2 = st.columns(2)

# Video stream placeholder in the first column
with col1:
    st.subheader("Video Stream")
    video_placeholder = st.empty()

# Results display in the second column
with col2:
    st.subheader("Detection Results")
    results_placeholder = st.empty()
    info_placeholder = st.empty()

# Start/Stop button for the stream
stream_active = st.sidebar.toggle("Activate Video Stream", value=False)

# Create a counter to skip frames for performance
frame_skip = st.sidebar.slider(
    "Process every N frames",
    min_value=1,
    max_value=10,
    value=DEFAULT_FRAME_SKIP,
    step=1,
)


# Display class counts and detection information
def display_detection_info(results):
    if len(results[0].boxes) > 0:
        # Get detection details
        boxes = results[0].boxes

        # Create a dictionary to count classes
        class_counts = {}

        for box in boxes:
            class_id = int(box.cls[0].item())
            class_name = results[0].names[class_id]

            # Update class counts
            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

        # Display summary
        detection_info = "### Detected Objects:\n"
        for class_name, count in class_counts.items():
            detection_info += f"- {class_name}: {count}\n"

        info_placeholder.markdown(detection_info)
    else:
        info_placeholder.write("No objects detected.")


# Main video stream loop
prev_stream_state = False

if stream_active != prev_stream_state:
    # Stream state has changed
    if not stream_active:
        # Stream was just deactivated
        if "cap" in st.session_state and st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        # Clear model from memory
        model = None
        clear_memory()
        # Reset placeholders
        video_placeholder.empty()
        results_placeholder.empty()
        info_placeholder.empty()
        st.sidebar.success("Stream stopped and resources released.")
    prev_stream_state = stream_active

if stream_active:
    try:
        # Load model only when stream is active
        with st.spinner("Loading YOLO model..."):
            model = load_model(model_version)
        st.sidebar.success(f"Model {model_version} loaded successfully!")

        # Initialize camera only when stream is active
        if "cap" not in st.session_state or st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(camera_id)
            cap = st.session_state.cap

            # Set resolution
            width, height = RESOLUTIONS[selected_resolution]
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Get actual resolution (may differ from requested)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if actual_width != width or actual_height != height:
                st.sidebar.warning(
                    f"Camera adjusted resolution to: {actual_width}x{actual_height}"
                )

            if not cap.isOpened():
                st.sidebar.error(f"Error: Could not open camera {camera_id}.")
                st.stop()

            # Display camera information
            st.sidebar.success(f"Connected to: {available_cameras[camera_id]}")
        else:
            cap = st.session_state.cap

        # Initialize frame counter
        frame_count = 0
        last_time = time.time()

        # Initialize heatmap object if heatmap mode is selected
        heatmap = None

        if detection_mode == "Heatmap":
            # Map colormap string to OpenCV constant
            colormap_dict = {
                "COLORMAP_JET": cv2.COLORMAP_JET,
                "COLORMAP_TURBO": cv2.COLORMAP_TURBO,
                "COLORMAP_HOT": cv2.COLORMAP_HOT,
                "COLORMAP_INFERNO": cv2.COLORMAP_INFERNO,
            }

            # Convert class options to class indices if specified
            classes = None
            if class_options:
                # Common COCO class mapping - adjust as needed

                classes = [
                    COCO_CLASS_NAMES.index(cls) if cls in COCO_CLASS_NAMES else None
                    for cls in class_options
                ]

            # Initialize the heatmap solution
            try:
                heatmap = solutions.Heatmap(
                    show=False,
                    model=model_version,
                    colormap=colormap_dict[heatmap_colormap],
                    conf=confidence_threshold,
                    classes=classes,
                )
            except Exception as e:
                st.sidebar.error(f"Error initializing heatmap: {e}")
                st.stop()

        # Initialize pose detection if pose mode is selected
        elif detection_mode == "Pose":
            try:
                # Use the model directly for pose estimation
                # The pose-specific models already have the right architecture
                # According to the documentation, we don't need PosePredictor
                # We'll just use the model directly and access the keypoints in the results
                st.sidebar.info("Using direct YOLO model for pose estimation")
            except Exception as e:
                st.sidebar.error(f"Error initializing pose detection: {e}")
                st.stop()

        # Flag to track heatmap reset
        need_reset = False

        # Create a stop button that will be checked in the loop
        stop_button = st.sidebar.button("Stop Stream")

        while stream_active and not stop_button:
            # Check if reset button was clicked (for heatmap mode)
            if detection_mode == "Heatmap" and reset_heatmap:
                need_reset = True

            # Read frame from webcam
            ret, frame = cap.read()

            if not ret:
                st.sidebar.error("Error: Failed to capture image from webcam.")
                break

            # Calculate FPS
            current_time = time.time()
            elapsed = current_time - last_time

            # Only process frames according to FPS limit
            if elapsed > 1.0 / fps_limit:
                # Reset heatmap if needed
                if need_reset and heatmap:
                    heatmap.reset_heatmap()
                    need_reset = False

                # Convert frame from BGR to RGB (for display)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Update frame counter
                frame_count += 1

                # Only process every n-th frame for performance
                if frame_count % frame_skip == 0:
                    try:
                        # Process frame based on selected detection mode
                        if detection_mode == "Standard":
                            # Run YOLO inference on the frame
                            results = model(
                                frame_rgb, conf=confidence_threshold, stream=True
                            )

                            # Process results generator and ensure it's exhausted
                            result = None
                            for r in results:
                                result = r  # Keep only the last result

                            if result is not None:
                                # Display the annotated image
                                annotated_img = result.plot()
                                results_placeholder.image(
                                    annotated_img,
                                    caption="Detection Results",
                                    use_container_width=True,
                                )
                                # Display detection information
                                display_detection_info([result])

                                # Clear result objects
                                del annotated_img
                                del result

                        elif detection_mode == "Heatmap":
                            # Process frame with heatmap
                            heatmap_result = heatmap(frame.copy())

                            if hasattr(heatmap_result, "plot_im") and isinstance(
                                heatmap_result.plot_im, np.ndarray
                            ):
                                heatmap_rgb = (
                                    cv2.cvtColor(
                                        heatmap_result.plot_im, cv2.COLOR_BGR2RGB
                                    )
                                    if heatmap_result.plot_im.ndim == 3
                                    else heatmap_result.plot_im
                                )
                                results_placeholder.image(
                                    heatmap_rgb,
                                    caption="Heatmap Results",
                                    use_container_width=True,
                                )
                                del heatmap_rgb

                            # Run separate detection for object counting
                            results = model(
                                frame_rgb, conf=confidence_threshold, stream=True
                            )
                            result = None
                            for r in results:
                                result = r

                            if result:
                                display_detection_info([result])
                                del result

                            del heatmap_result

                        elif detection_mode == "Segmentation":
                            # Process frame with segmentation
                            seg_results = model(
                                frame_rgb,
                                conf=seg_confidence,
                                show=False,
                                stream=True,
                            )

                            result = None
                            for r in seg_results:
                                result = r

                            if result is not None:
                                seg_img = result.plot(
                                    boxes=show_boxes, masks=show_masks
                                )
                                results_placeholder.image(
                                    seg_img,
                                    caption="Segmentation Results",
                                    use_container_width=True,
                                )

                                # Extract and display segmentation information
                                seg_info = "### Segmentation Detection\n"

                                if (
                                    hasattr(result, "masks")
                                    and result.masks is not None
                                ):
                                    num_masks = len(result.masks)
                                    seg_info += f"- Objects with masks: {num_masks}\n"

                                    class_counts = {}
                                    for box in result.boxes:
                                        class_id = int(box.cls[0].item())
                                        class_name = result.names[class_id]
                                        class_counts[class_name] = (
                                            class_counts.get(class_name, 0) + 1
                                        )

                                    seg_info += "### Segmented Objects:\n"
                                    for class_name, count in class_counts.items():
                                        seg_info += f"- {class_name}: {count}\n"
                                else:
                                    seg_info += "- No segmented objects detected\n"

                                info_placeholder.markdown(seg_info)
                                del seg_img
                                del result

                        elif detection_mode == "Pose":
                            # Process frame with pose detection
                            pose_results = model(
                                frame_rgb,
                                conf=pose_confidence,
                                show=False,
                                stream=True,
                            )

                            result = None
                            for r in pose_results:
                                result = r

                            if result is not None:
                                pose_img = result.plot(
                                    kpt_line=show_skeleton,
                                    kpt_radius=(3 if show_keypoints else 0),
                                    conf=pose_confidence,
                                )
                                results_placeholder.image(
                                    pose_img,
                                    caption="Pose Detection Results",
                                    use_container_width=True,
                                )

                                pose_info = "### Pose Detection\n"
                                if (
                                    hasattr(result, "keypoints")
                                    and result.keypoints is not None
                                ):
                                    num_people = len(result.boxes)
                                    pose_info += f"- People detected: {num_people}\n"
                                    if num_people > 0 and hasattr(
                                        result.keypoints, "xy"
                                    ):
                                        total_keypoints = result.keypoints.xy.shape[1]
                                        pose_info += f"- Keypoints per person: {total_keypoints}\n"
                                else:
                                    pose_info += "- No poses detected\n"

                                info_placeholder.markdown(pose_info)
                                del pose_img
                                del result

                    finally:
                        # Clear memory after processing each frame
                        clear_memory()

                # Always display the camera feed
                video_placeholder.image(
                    frame_rgb, caption="Video Stream", use_container_width=True
                )
                del frame_rgb

                # Update FPS timing
                last_time = current_time

            # Check if stop button was clicked
            if stop_button:
                break

            # Add a small sleep to prevent high CPU usage
            time.sleep(0.01)

    except Exception as e:
        st.sidebar.error(f"Error in video stream: {str(e)}")
    finally:
        # Clear resources if there was an error
        if not stream_active:
            if "cap" in st.session_state and st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            clear_memory()
            model = None
else:
    # Display a message when the stream is not active
    video_placeholder.info(
        "Click 'Activate Video Stream' in the sidebar to start the webcam."
    )
    results_placeholder.info(
        "Detection results will appear here once the stream is active."
    )
    # Ensure camera is released when stream is inactive
    if "cap" in st.session_state and st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
