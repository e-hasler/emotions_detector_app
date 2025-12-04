import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import numpy as np
import av
import mediapipe as mp
from threading import Lock
import logging
# ADDED 'os' for creating absolute paths
import os 
from tensorflow.keras.models import load_model

# Suppress the ScriptRunContext warnings
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)

# --- Configuration ---
CONFIDENCE_THRESHOLD = 0.5
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'] 
PASSWORD = st.secrets.get("TURN_PASSWORD", "")

# --- WebRTC Configuration for Robustness (Fixes Cloud Deployment) ---

RTC_CONFIGURATION = {
    "iceServers": [
        # Public STUN servers
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun.services.mozilla.com"]},
        
        {
            "urls": "turn:global.relay.metered.ca:80", 
            "username": "f5886a41b6f59a824c65c241",
            "credential": PASSWORD
        }
    ]
}

# The rest of the file remains the same...

# --- Load Emotion Model ---
@st.cache_resource
def load_emotion_model():
    """Load the emotion detection model using an absolute path."""
    # Get the absolute path to the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'original_cnn_best.keras')
    
    try:
        # Use the full, absolute path for guaranteed file loading
        model = load_model(model_path)
        print(f"Emotion detection model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        # This will now show the absolute path that failed, which is great for debugging
        print(f"Error loading model from {model_path}: {e}")
        return None

emotion_model = load_emotion_model()

class GlobalFrameStore:
    def __init__(self):
        self.frame = None
        self.lock = Lock()
    
    def set_frame(self, frame):
        with self.lock:
            self.frame = frame.copy() if frame is not None else None
    
    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

# Use st.cache_resource to persist across reruns
@st.cache_resource
def get_frame_store():
    print("Creating new GlobalFrameStore instance")
    return GlobalFrameStore()

global_frame_store = get_frame_store()

# --- Session State Initialization ---
if 'display_frame' not in st.session_state:
    st.session_state.display_frame = None

if 'captured_emotion' not in st.session_state:
    st.session_state.captured_emotion = None

if 'captured_confidence' not in st.session_state:
    st.session_state.captured_confidence = None

if 'capture_message' not in st.session_state:
    st.session_state.capture_message = ""

if "last_bbox" not in st.session_state:
    st.session_state.last_bbox = None

# --- MediaPipe Face Detection Setup ---
@st.cache_resource
def get_face_detector():
    """Caches the MediaPipe detector to prevent re-loading on every stream rerun."""
    mp_face_detection = mp.solutions.face_detection
    face_detector = mp_face_detection.FaceDetection(
        model_selection=0,  # 0 = short-range (webcam optimized)
        min_detection_confidence=CONFIDENCE_THRESHOLD
    )
    return face_detector, mp_face_detection

# --- Video Processor ---
class FaceDetector(VideoProcessorBase): 
    def __init__(self):
        self.face_detector, self.mp_face_detection = get_face_detector()
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24") 
        
        # Store frame in global variable
        global_frame_store.set_frame(img)

        img_h, img_w, _ = img.shape
        
        # Convert BGR to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detect faces 
        results = self.face_detector.process(img_rgb)
        
        # --- Draw Detection Boxes (on the live stream) ---
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                
                x = int(bboxC.xmin * img_w)
                y = int(bboxC.ymin * img_h)
                w = int(bboxC.width * img_w)
                h = int(bboxC.height * img_h)
                
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                confidence = detection.score[0] * 100
                label = f"Face: {confidence:.1f}%"
                
                cv2.putText(img, label, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Save last detected bounding box (scaled to pixel coords)
                st.session_state.last_bbox = (x, y, w, h)

        
        # Convert back to VideoFrame for the stream display
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit App ---
st.title("📸 Real-Time Face Detection")
st.write("Detection uses MediaPipe BlazeFace. Click 'Click to analyse' to capture the image and display it below.")

# --- Callback function definition (Runs in Main Thread) ---
def analyse_face():
    """Captures the last known frame, converts it to RGB, saves it for display, and updates status."""
    
    # Get the latest frame from global storage
    captured_frame = global_frame_store.get_frame()
    
    if captured_frame is not None and captured_frame.size > 0:
        # Convert the captured BGR frame to RGB format for correct Streamlit display
        captured_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
        st.session_state.display_frame = captured_rgb
        st.session_state.capture_message = "✅ Image captured successfully! See the result below."
    else:
        st.session_state.display_frame = None
        st.session_state.capture_message = "⚠️ Could not capture a valid frame. Please ensure the webcam stream is active and try again."

# --- Layout ---
st.subheader("Live Camera Stream")
webrtc_streamer(
    key="face-detection",
    video_processor_factory=FaceDetector, 
    # Use the robust configuration
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- Button with callback ---
st.button("Click to analyse", type="primary", on_click=analyse_face) 

# --- Display Capture Message ---
if st.session_state.capture_message:
    if st.session_state.capture_message.startswith("✅"):
        st.success(st.session_state.capture_message)
    else:
        st.warning(st.session_state.capture_message)

# --- Captured Image Display ---
if st.session_state.display_frame is not None:
    st.subheader("Captured Image")
    
    # Create a copy to draw on
    display_img = st.session_state.display_frame.copy()
    h, w = display_img.shape[:2]


    # ------------ ANALYSE EMOTION OF THE CAPTURED IMAGE display_img ------------------------
    
    # CHECK IF MODEL IS LOADED BEFORE CALLING .predict() (FIXES THE ATTRIBUTE ERROR)
    if emotion_model is not None:
        try:
            # Preprocess the image for the emotion model (Corrected: No Grayscale)
            processed_face = cv2.resize(display_img, (48, 48))
            processed_face = processed_face.astype("float32") / 255.0
            # Add batch dimension -> Shape becomes (1, 48, 48, 3)
            processed_face = np.expand_dims(processed_face, axis=0)
            
            # Predict emotion
            predictions = emotion_model.predict(processed_face, verbose=0)[0]
            emotion_idx = np.argmax(predictions)
            emotion_label = EMOTION_LABELS[emotion_idx]
            emotion_confidence = np.max(predictions) * 100
            
            # Update session state with results
            st.session_state.captured_emotion = emotion_label
            st.session_state.captured_confidence = emotion_confidence
            
            print(f"Emotion: {emotion_label}, Confidence: {emotion_confidence:.2f}%")
        
        except Exception as e:
            # Catch potential errors during prediction (e.g., shape mismatch)
            st.error(f"Prediction failed with error: {e}")
            print(f"Prediction Error: {e}")
            st.session_state.captured_emotion = None
            st.session_state.captured_confidence = None
    
    else:
        # Display an error if the model failed to load
        st.error("⚠️ Emotion model failed to load. Check your terminal for details.")
        st.session_state.captured_emotion = None
        st.session_state.captured_confidence = None

    # ------------ END OF EMOTION ANALYSIS ------------------------
    
    # Draw green box around the image border (not the detected face)
    border_thickness = 3
    cv2.rectangle(display_img, (border_thickness, border_thickness), 
                  (w - border_thickness, h - border_thickness), 
                  (0, 255, 0), border_thickness)
    
    # Display emotion if available
    if st.session_state.captured_emotion is not None:
        emotion_text = f"{st.session_state.captured_emotion.upper()}: {st.session_state.captured_confidence:.1f}%"
        
        # Put text with background for better visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(emotion_text, font, font_scale, font_thickness)
        
        # Draw background rectangle
        cv2.rectangle(display_img, (10, 10), (20 + text_width, 20 + text_height + baseline), (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(display_img, emotion_text, (15, 15 + text_height), 
                    font, font_scale, (0, 0, 0), font_thickness)
    
    st.image(display_img, caption="Analyzed Frame", width='stretch')

st.markdown("---")
st.markdown("""
### How it works:
- **Green boxes** appear around detected faces
- **Confidence scores** show detection certainty
- Works in real-time with continuous video stream
- Pressing **'Click to analyse'** captures the current frame and displays it below with an emotion prediction.
""")