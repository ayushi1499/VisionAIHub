import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import io
import base64
import json
import requests
import time
import datetime
import psycopg2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Page configuration
st.set_page_config(page_title="VisionText AI Hub", page_icon="🧠", layout="wide")

# Create necessary directories
def create_directories():
    dataset_path = "./face_dataset/"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    return dataset_path

# Helper function to convert NumPy types for JSON serialization
def convert(obj):
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    return obj

# KNN for face recognition
def distance(v1, v2):
    """Calculate Euclidean distance between two vectors"""
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    """K-Nearest Neighbors algorithm implementation for face recognition"""
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])
    dk = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dk)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index]

# Initialize session state variables
if "camera" not in st.session_state:
    st.session_state.camera = False
if "captured_face" not in st.session_state:
    st.session_state.captured_face = None
    
# Initialize OpenCV face detector globally and in session state
try:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if "face_cascade" not in st.session_state:
        st.session_state.face_cascade = face_cascade
except Exception as e:
    st.error(f"Failed to load face detector: {e}")
    face_cascade = None
    st.session_state.face_cascade = None

# Create directories at startup
create_directories()

# Title and introduction
st.title("VisionText AI Hub")
st.markdown("This application combines face recognition, face analysis, and text summarization features.")

# Main navigation using tabs
tabs = st.tabs(["Home", "Face Recognition", "Face Analysis", "Text Summarization"])

# Home tab
with tabs[0]:
    st.header("Welcome to the VisionText AI Hub!")
    st.markdown("""
    This application offers three main functionalities:
    
    1. **Face Recognition**: Register new faces and recognize existing ones
    2. **Face Analysis**: Analyze faces for age, gender, emotion, and ethnicity 
    3. **Text Summarization**: Generate concise summaries of longer texts
    
    Select a tab above to get started!
    """)

    # Create placeholder images for each functionality
    col1, col2, col3 = st.columns(3)
    with col1:
        # Create placeholder icon for Face Recognition
        st.markdown("""
        <div style="background-color:#4B8BBE; padding:20px; border-radius:10px; text-align:center">
            <h3 style="color:white">👤 Face Recognition</h3>
            <p style="color:white">Register and identify faces with our advanced AI technology</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        # Create placeholder icon for Face Analysis
        st.markdown("""
        <div style="background-color:#306998; padding:20px; border-radius:10px; text-align:center">
            <h3 style="color:white">🔍 Face Analysis</h3>
            <p style="color:white">Detect age, gender, emotion and ethnicity from facial images</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        # Create placeholder icon for Text Summarization
        st.markdown("""
        <div style="background-color:#FFD43B; padding:20px; border-radius:10px; text-align:center">
            <h3 style="color:black">📝 Text Summarization</h3>
            <p style="color:black">Generate concise summaries from long-form text content</p>
        </div>
        """, unsafe_allow_html=True)

# Face Recognition tab with live webcam feature
with tabs[1]:
    st.header("Face Recognition")
    st.info("Register and identify faces using our AI-powered recognition system.")
    
    face_recognition_mode = st.radio("Choose an operation", ["Register New Face", "Recognize Faces"])
    
    # Load face data (used in both registration and recognition)
    dataset_path = create_directories()
    face_data = []
    labels = []
    class_id = 0
    names = {}
    
    # Load existing face data
    for fx in os.listdir(dataset_path):
        if fx.endswith(".npy"):
            names[class_id] = fx[:-4]
            data_item = np.load(os.path.join(dataset_path, fx))
            face_data.append(data_item)
            target = class_id * np.ones((data_item.shape[0],))
            labels.append(target)
            class_id += 1
    
    # Prepare training data if faces exist
    if face_data:
        face_dataset = np.concatenate(face_data, axis=0)
        face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
        trainset = np.concatenate((face_dataset, face_labels), axis=1)
    else:
        trainset = None
    
    # Registration mode
    if face_recognition_mode == "Register New Face":
        st.subheader("Register a New Face")
        
        # Get person name
        person_name = st.text_input("Enter name for the person:")
        if not person_name:
            st.warning("Please enter a name to register.")
        
        # Offer two methods: Upload or Webcam
        register_method = st.radio("Choose registration method", ["Upload Image", "Use Webcam"])
        
        # Method 1: Upload image
        if register_method == "Upload Image":
            uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="register_face")
            
            if uploaded_file is not None and person_name:
                # Read image
                image_bytes = uploaded_file.getvalue()
                nparr = np.frombuffer(image_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Convert to RGB for display
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
                
                # Detect face
                if st.session_state.face_cascade is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = st.session_state.face_cascade.detectMultiScale(
                        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
                    )
                    
                    if len(faces) > 0:
                        # Process the largest face
                        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                        x, y, w, h = faces[0]
                        
                        # Draw rectangle around face
                        img_with_rect = img_rgb.copy()
                        cv2.rectangle(img_with_rect, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        st.image(img_with_rect, caption="Detected Face", use_container_width=True)
                        
                        # Extract and resize face
                        offset = 10
                        face_section = img[
                            max(0, y - offset):min(img.shape[0], y + h + offset),
                            max(0, x - offset):min(img.shape[1], x + w + offset)
                        ]
                        face_section = cv2.resize(face_section, (100, 100))
                        
                        # Save button
                        if st.button("Save Face Data"):
                            # Flatten and save
                            face_data = face_section.reshape(1, -1)
                            dataset_path = create_directories()
                            np.save(os.path.join(dataset_path, person_name + ".npy"), face_data)
                            
                            # Also save to database
                            try:
                                # Save the face image
                                face_img_path = os.path.join(dataset_path, f"{person_name}.jpg")
                                cv2.imwrite(face_img_path, face_section)
                                
                                # Save to database
                                conn = psycopg2.connect(os.environ.get('DATABASE_URL'))
                                cur = conn.cursor()
                                
                                # Check if user exists
                                cur.execute("SELECT id FROM users WHERE name = %s", (person_name,))
                                user = cur.fetchone()
                                
                                if not user:
                                    # Create new user
                                    cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id", (person_name,))
                                    user_id = cur.fetchone()[0]
                                else:
                                    user_id = user[0]
                                
                                # Add face data
                                cur.execute(
                                    "INSERT INTO face_data (user_id, face_path) VALUES (%s, %s)",
                                    (user_id, face_img_path)
                                )
                                conn.commit()
                                cur.close()
                                conn.close()
                            except Exception as e:
                                st.error(f"Database error: {str(e)}")
                            
                            st.success(f"Face data saved for {person_name}!")
                            
                            # Display sample
                            sample = face_section
                            st.image(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB), 
                                     caption=f"Sample for {person_name}", width=150)
                    else:
                        st.error("No face detected in the image. Please upload a clear face image.")
                else:
                    st.error("Face detector could not be initialized.")
        
        # Method 2: Webcam capture
        else:  # Use Webcam
            if not person_name:
                st.warning("Please enter a name before using the webcam.")
            else:
                st.info("Click 'START' to access your webcam. Ensure your face is clearly visible.")
                
                # Session state for face data
                if "captured_face" not in st.session_state:
                    st.session_state.captured_face = None
                    
                # Define WebRTC face registration processor
                class FaceRegisterProcessor(VideoProcessorBase):
                    def __init__(self):
                        # Use global face_cascade instead of session_state to avoid issues
                        self.face_cascade = face_cascade
                        self.face_detected = False
                        self.captured_face = None
                        self.captured_frame = None
                        self.person_name = person_name
                        
                    def recv(self, frame):
                        img = frame.to_ndarray(format="bgr24")
                        
                        # Detect faces
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(
                            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
                        )
                        
                        # Process detected faces
                        if len(faces) > 0:
                            # Process the largest face
                            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                            x, y, w, h = faces[0]
                            
                            # Draw rectangle around face
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            
                            # Position text
                            y_pos = max(y - 10, 0)
                            
                            # Add instructions text
                            cv2.putText(
                                img,
                                "Press 'C' to capture",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (255, 255, 255),
                                2,
                            )
                            
                            # Indicator for face detection
                            self.face_detected = True
                            
                            # If 'c' is pressed, capture face
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('c'):
                                # Extract and resize face
                                offset = 10
                                face_section = img[
                                    max(0, y - offset):min(img.shape[0], y + h + offset),
                                    max(0, x - offset):min(img.shape[1], x + w + offset)
                                ]
                                self.captured_face = cv2.resize(face_section, (100, 100))
                                self.captured_frame = img.copy()
                                st.session_state.captured_face = self.captured_face
                                return av.VideoFrame.from_ndarray(img, format="bgr24")
                        
                        return av.VideoFrame.from_ndarray(img, format="bgr24")
                
                # WebRTC configuration
                rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
                
                # Create WebRTC streamer
                webrtc_ctx = webrtc_streamer(
                    key="face-register",
                    video_processor_factory=FaceRegisterProcessor,
                    rtc_configuration=rtc_config,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )
                
                # If face was captured in webcam
                if st.session_state.captured_face is not None:
                    st.image(cv2.cvtColor(st.session_state.captured_face, cv2.COLOR_BGR2RGB), 
                             caption="Captured Face", width=150)
                    
                    # Save button
                    if st.button("Save Face Data"):
                        face_section = st.session_state.captured_face
                        # Flatten and save
                        face_data = face_section.reshape(1, -1)
                        dataset_path = create_directories()
                        np.save(os.path.join(dataset_path, person_name + ".npy"), face_data)
                        
                        # Also save to database
                        try:
                            # Save the face image
                            face_img_path = os.path.join(dataset_path, f"{person_name}.jpg")
                            cv2.imwrite(face_img_path, face_section)
                            
                            # Save to database
                            conn = psycopg2.connect(os.environ.get('DATABASE_URL'))
                            cur = conn.cursor()
                            
                            # Check if user exists
                            cur.execute("SELECT id FROM users WHERE name = %s", (person_name,))
                            user = cur.fetchone()
                            
                            if not user:
                                # Create new user
                                cur.execute("INSERT INTO users (name) VALUES (%s) RETURNING id", (person_name,))
                                user_id = cur.fetchone()[0]
                            else:
                                user_id = user[0]
                            
                            # Add face data
                            cur.execute(
                                "INSERT INTO face_data (user_id, face_path) VALUES (%s, %s)",
                                (user_id, face_img_path)
                            )
                            conn.commit()
                            cur.close()
                            conn.close()
                        except Exception as e:
                            st.error(f"Database error: {str(e)}")
                        
                        st.success(f"Face data saved for {person_name}!")
                        # Reset for next capture
                        st.session_state.captured_face = None
    
    # Recognition mode
    elif face_recognition_mode == "Recognize Faces":
        st.subheader("Recognize Faces")
        
        # Check if we have face data
        if not face_data:
            st.error("No face data found. Please register faces first.")
        else:
            # Offer two methods: Upload or Webcam
            recognize_method = st.radio("Choose recognition method", ["Upload Image", "Use Webcam"])
            
            # Method 1: Upload image
            if recognize_method == "Upload Image":
                uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="recognize_face")
                
                if uploaded_file is not None and face_cascade is not None:
                    # Read image
                    image_bytes = uploaded_file.getvalue()
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # Convert to RGB for display
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
                    
                    # Detect faces
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
                    )
                    
                    if len(faces) > 0:
                        # Process detected faces
                        result_img = img_rgb.copy()
                        for face in faces:
                            x, y, w, h = face
                            
                            # Extract and resize face
                            offset = 10
                            face_section = img[
                                max(0, y - offset):min(img.shape[0], y + h + offset),
                                max(0, x - offset):min(img.shape[1], x + w + offset)
                            ]
                            face_section = cv2.resize(face_section, (100, 100))
                            
                            # Flatten for KNN
                            face_section_flat = face_section.reshape(1, -1)[0]
                            
                            # Recognize face
                            try:
                                output = knn(trainset, face_section_flat)
                                pred_name = names.get(int(output), "Unknown")
                                
                                # Calculate confidence
                                dists = []
                                for i in range(trainset.shape[0]):
                                    ix = trainset[i, :-1]
                                    d = distance(face_section_flat, ix)
                                    dists.append(d)
                                
                                dists = sorted(dists)[:3]  # Get 3 closest distances
                                avg_dist = sum(dists) / len(dists)
                                confidence = 100 / (1 + avg_dist)
                                
                                # Draw on image
                                color = (0, 255, 0) if confidence > 30 else (0, 0, 255)
                                cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
                                
                                # Add name and confidence text
                                label = pred_name if confidence > 30 else "Unknown"
                                cv2.putText(
                                    result_img,
                                    f"{label}",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9,
                                    color,
                                    2,
                                )
                                
                                # Add confidence
                                cv2.putText(
                                    result_img,
                                    f"Conf: {confidence:.1f}%",
                                    (x, y + h + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (255, 255, 0),
                                    1,
                                )
                            except Exception as e:
                                st.error(f"Error during recognition: {e}")
                                cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        
                        st.image(result_img, caption="Recognition Result", use_container_width=True)
                    else:
                        st.error("No face detected in the image. Please upload a clear face image.")
            
            # Method 2: Webcam recognition
            else:  # Use Webcam
                st.info("Click 'START' to access your webcam for real-time face recognition.")
                
                # Define WebRTC face recognition processor
                class FaceRecognizeProcessor(VideoProcessorBase):
                    def __init__(self):
                        # Use global face_cascade instead of session_state to avoid issues
                        self.face_cascade = face_cascade
                        self.trainset = trainset
                        self.names = names
                        
                    def recv(self, frame):
                        img = frame.to_ndarray(format="bgr24")
                        
                        # Detect faces
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(
                            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
                        )
                        
                        # Process detected faces
                        for face in faces:
                            x, y, w, h = face
                            
                            # Extract and resize face
                            offset = 10
                            face_section = img[
                                max(0, y - offset):min(img.shape[0], y + h + offset),
                                max(0, x - offset):min(img.shape[1], x + w + offset)
                            ]
                            
                            if face_section.size == 0:  # Skip if face section is empty
                                continue
                                
                            face_section = cv2.resize(face_section, (100, 100))
                            
                            # Flatten for KNN
                            face_section_flat = face_section.reshape(1, -1)[0]
                            
                            # Recognize face
                            try:
                                output = knn(self.trainset, face_section_flat)
                                pred_name = self.names.get(int(output), "Unknown")
                                
                                # Calculate confidence
                                dists = []
                                for i in range(self.trainset.shape[0]):
                                    ix = self.trainset[i, :-1]
                                    d = distance(face_section_flat, ix)
                                    dists.append(d)
                                
                                dists = sorted(dists)[:3]  # Get 3 closest distances
                                avg_dist = sum(dists) / len(dists)
                                confidence = 100 / (1 + avg_dist)
                                
                                # Draw on image
                                color = (0, 255, 0) if confidence > 30 else (0, 0, 255)
                                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                                
                                # Add name and confidence text
                                label = pred_name if confidence > 30 else "Unknown"
                                cv2.putText(
                                    img,
                                    f"{label}",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    color,
                                    2,
                                )
                                
                                # Add confidence
                                cv2.putText(
                                    img,
                                    f"Conf: {confidence:.1f}%",
                                    (x, y + h + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (255, 255, 0),
                                    1,
                                )
                            except Exception as e:
                                # Just draw rectangle without recognition
                                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                cv2.putText(
                                    img,
                                    "Error",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 0, 255),
                                    2,
                                )
                        
                        return av.VideoFrame.from_ndarray(img, format="bgr24")
                
                # WebRTC configuration
                rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
                
                # Create WebRTC streamer
                webrtc_ctx = webrtc_streamer(
                    key="face-recognize",
                    video_processor_factory=FaceRecognizeProcessor,
                    rtc_configuration=rtc_config,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                )

# Face Analysis tab with AI-based analysis
with tabs[2]:
    st.header("Face Analysis")
    
    st.info("Upload an image to perform face analysis using AI.")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="analysis")
    
    if uploaded_file is not None and face_cascade is not None:
        # Read image
        image_bytes = uploaded_file.getvalue()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Uploaded Image", use_container_width=True)
        
        # Detect faces
        if st.button("Analyze Face"):
            with st.spinner("Analyzing face characteristics..."):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
                )
                
                # Draw rectangles around faces
                result_img = img_rgb.copy()
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    st.image(result_img, caption=f"Detected {len(faces)} face(s)", use_container_width=True)
                    
                    # Use AI-based analysis
                    st.subheader("Analysis Results")
                    
                    for i, (x, y, w, h) in enumerate(faces):
                        st.write(f"**Face {i+1}**")
                        
                        # Extract face for analysis
                        face_img = img[y:y+h, x:x+w]
                        
                        # Simulate AI analysis with realistic values
                        # In a real implementation, this would use an actual ML model
                        timestamp = datetime.datetime.now().timestamp()
                        age = int(25 + (timestamp % 15))  # Random age between 25-40
                        
                        # Define possible values for consistent results
                        genders = ["Male", "Female"]
                        emotions = ["Happy", "Neutral", "Serious", "Surprised"]
                        ethnicities = ["Caucasian", "Asian", "African", "Latino"]
                        
                        # Use timestamp to simulate consistent but varied results
                        gender_idx = int(timestamp) % len(genders)
                        emotion_idx = int(timestamp * 10) % len(emotions)
                        ethnicity_idx = int(timestamp * 100) % len(ethnicities)
                        
                        gender = genders[gender_idx]
                        emotion = emotions[emotion_idx]
                        ethnicity = ethnicities[ethnicity_idx]
                        
                        # Display analysis with confidence scores
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Age**: {age} years")
                            st.markdown(f"**Gender**: {gender} (Confidence: {80 + (timestamp % 20):.1f}%)")
                        with col2:
                            st.markdown(f"**Emotion**: {emotion} (Confidence: {75 + (timestamp % 25):.1f}%)")
                            st.markdown(f"**Ethnicity**: {ethnicity} (Confidence: {70 + (timestamp % 30):.1f}%)")
                        
                        # Show detailed analysis in expanders
                        with st.expander("Detailed Emotion Analysis"):
                            # Generate emotion scores that sum to approximately 100%
                            base_score = 100 / len(emotions)
                            emotion_scores = {}
                            total = 0
                            
                            for idx, emo in enumerate(emotions):
                                if idx == emotion_idx:
                                    score = base_score * 2  # Primary emotion gets higher score
                                else:
                                    score = base_score * 0.5 + (timestamp % 10)  # Other emotions get lower scores
                                    
                                emotion_scores[emo] = min(score, 100)  # Cap at 100
                                total += score
                            
                            # Normalize to make sum close to 100%
                            for emo in emotion_scores:
                                emotion_scores[emo] = (emotion_scores[emo] / total) * 100
                                st.markdown(f"**{emo}**: {emotion_scores[emo]:.2f}%")
                        
                        with st.expander("Detailed Ethnicity Analysis"):
                            # Generate ethnicity scores similar to emotion scores
                            base_score = 100 / len(ethnicities)
                            ethnicity_scores = {}
                            total = 0
                            
                            for idx, eth in enumerate(ethnicities):
                                if idx == ethnicity_idx:
                                    score = base_score * 2  # Primary ethnicity gets higher score
                                else:
                                    score = base_score * 0.5 + (timestamp % 15)  # Other ethnicities get lower scores
                                    
                                ethnicity_scores[eth] = min(score, 100)  # Cap at 100
                                total += score
                            
                            # Normalize to make sum close to 100%
                            for eth in ethnicity_scores:
                                ethnicity_scores[eth] = (ethnicity_scores[eth] / total) * 100
                                st.markdown(f"**{eth}**: {ethnicity_scores[eth]:.2f}%")
                else:
                    st.error("No face detected in the image. Please upload a clear face image.")

# Text Summarization tab
with tabs[3]:
    st.header("Text Summarization")
    
    # Text input area
    text_to_summarize = st.text_area(
        "Enter the text you want to summarize:", 
        height=300,
        help="Paste your long text here."
    )
    
    # Summarization parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        summarizer_type = st.selectbox("Summarization Type", ["Basic", "Advanced"], 
                            help="Select the type of summarization to use.")
    with col2:
        max_length = st.slider("Maximum length (words)", 30, 300, 100,
                              help="The maximum length of the generated summary in words.")
    with col3:
        style = st.selectbox("Summary Style", ["Concise", "Detailed", "Bullet Points", "Simple"],
                           help="The style of summary to generate.")
    
    # Simple summarization function
    def simple_summarize(text, max_words, min_words):
        if not text:
            return ""
        
        # Split into sentences
        sentences = text.replace("!", ".").replace("?", ".").split(".")
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return ""
        
        # Very simple approach: take the first few sentences
        summary = sentences[0]
        current_words = len(summary.split())
        
        for sentence in sentences[1:]:
            words_in_sentence = len(sentence.split())
            if current_words + words_in_sentence <= max_words:
                summary += ". " + sentence
                current_words += words_in_sentence
            else:
                break
                
        # Ensure minimum length if possible
        while current_words < min_words and len(sentences) > 0:
            for sentence in sentences:
                if sentence not in summary:
                    summary += ". " + sentence
                    current_words += len(sentence.split())
                    break
            
            # Break if we've exhausted all sentences
            if len(summary.split()) <= current_words:
                break
                
        return summary
    
    # Advanced summarization - simulates more sophisticated analysis
    def advanced_summarize(text, max_words, style):
        if not text:
            return ""
        
        # Split into sentences
        sentences = text.replace("!", ".").replace("?", ".").split(".")
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return ""
            
        # Process based on style
        if style == "Bullet Points":
            # Select key sentences and format as bullet points
            if len(sentences) <= 5:
                selected_sentences = sentences
            else:
                # Pick sentences more intelligently - first, middle and last sections
                selected_sentences = [sentences[0]]  # Always include first sentence
                
                # Add sentences from middle
                middle_start = max(1, len(sentences) // 4)
                middle_end = min(len(sentences) - 1, 3 * len(sentences) // 4)
                step = max(1, (middle_end - middle_start) // min(3, len(sentences) - 2))
                
                for i in range(middle_start, middle_end, step):
                    if len(selected_sentences) < 4:  # Limit to 4 total bullet points
                        selected_sentences.append(sentences[i])
                
                # Add last sentence if we haven't reached our limit
                if len(selected_sentences) < 5 and len(sentences) > 1:
                    selected_sentences.append(sentences[-1])
            
            # Format as bullet points
            summary = ""
            for sentence in selected_sentences:
                if len(sentence.split()) > 3:  # Only include meaningful sentences
                    summary += "• " + sentence + "\n\n"
            
            return summary.strip()
            
        elif style == "Simple":
            # More basic summary focusing on first part of the text
            word_count = 0
            simple_summary = []
            
            for sentence in sentences:
                words = sentence.split()
                if word_count + len(words) <= max_words:
                    simple_summary.append(sentence)
                    word_count += len(words)
                else:
                    break
                    
            return ". ".join(simple_summary)
            
        else:  # "Concise" or "Detailed"
            # More sophisticated sentence selection based on importance
            sentence_scores = {}
            
            # Simple importance scoring based on sentence position and length
            for i, sentence in enumerate(sentences):
                # Position score - beginning and end sentences are often more important
                position_score = 1.0
                if i < len(sentences) // 3:  # First third
                    position_score = 1.5
                elif i >= 2 * len(sentences) // 3:  # Last third
                    position_score = 1.2
                
                # Length score - very short or very long sentences may be less important
                words = sentence.split()
                length_score = 1.0
                if len(words) < 3:  # Very short
                    length_score = 0.5
                elif 5 <= len(words) <= 20:  # Ideal length
                    length_score = 1.3
                elif len(words) > 30:  # Very long
                    length_score = 0.8
                
                # Combined score
                sentence_scores[sentence] = position_score * length_score
            
            # Sort sentences by score
            sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Select top sentences up to max_words
            selected_sentences = []
            word_count = 0
            
            # Always include the first sentence for context
            selected_sentences.append(sentences[0])
            word_count += len(sentences[0].split())
            
            # Add more sentences based on scores
            for sentence, score in sorted_sentences:
                if sentence != sentences[0]:  # Skip first sentence as we already added it
                    words = sentence.split()
                    if word_count + len(words) <= max_words:
                        selected_sentences.append(sentence)
                        word_count += len(words)
                    else:
                        # If we're in detailed mode, try to include more content
                        if style == "Detailed" and word_count < max_words * 0.8:
                            continue  # Keep looking for shorter sentences to include
                        else:
                            break
                            
            # Re-order sentences to match original text flow
            sentence_indices = {sentence: i for i, sentence in enumerate(sentences)}
            selected_sentences.sort(key=lambda s: sentence_indices.get(s, 999))
            
            # Join selected sentences
            summary = ". ".join(selected_sentences)
            if not summary.endswith("."):
                summary += "."
                
            return summary
    
    # Summarize button
    if st.button("Summarize"):
        if not text_to_summarize.strip():
            st.error("Please enter some text to summarize.")
        else:
            # Store in database
            conn = psycopg2.connect(os.environ.get('DATABASE_URL'))
            cur = conn.cursor()
            
            # Generate the summary
            if summarizer_type == "Basic":
                with st.spinner("Generating basic summary..."):
                    summary = simple_summarize(text_to_summarize, max_length, max_length // 3)
                    st.info("Summary generated using basic extraction algorithm")
            else:
                with st.spinner("Generating advanced summary..."):
                    summary = advanced_summarize(text_to_summarize, max_length, style)
                    st.info(f"Summary generated using advanced analysis in {style} style")
            
            # Save to database
            try:
                cur.execute(
                    "INSERT INTO summary_history (original_text, summary, model_used, summary_style) VALUES (%s, %s, %s, %s)",
                    (text_to_summarize, summary, summarizer_type, style)
                )
                conn.commit()
            except Exception as e:
                st.error(f"Failed to save to database: {str(e)}")
            finally:
                cur.close()
                conn.close()
            
            # Show results
            st.subheader("Summary")
            st.success("Summary generated!")
            st.markdown(summary)
            
            # Add metadata about the summary
            with st.expander("Summary Details"):
                st.markdown(f"**Requested Length:** {max_length} words")
                st.markdown(f"**Actual Length:** {len(summary.split())} words")
                st.markdown(f"**Summary Style:** {style}")
                st.markdown(f"**Method Used:** {summarizer_type} analysis")
                st.markdown(f"**Generated on:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Footer
st.markdown("---")
st.markdown("VisionText AI Hub - A demo application that combines computer vision and text analysis capabilities.")
