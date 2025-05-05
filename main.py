import glob
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from datetime import datetime
import pandas as pd
from transformers import pipeline
from deepface import DeepFace
import tempfile
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import shutil
import random


# Page configuration
st.set_page_config(page_title="VisionText AI Hub", page_icon="🧠", layout="wide")

# Apply custom CSS styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4285F4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0f9d58;
        margin-bottom: 1rem;
    }
    .section {
        background-color: #f5f5f5;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and introduction
st.markdown(
    "<h1 class='main-header'>VisionText AI Hub</h1>", unsafe_allow_html=True
)
st.markdown(
    "This application combines face recognition, face analysis, and text summarization features."
)

# Create necessary directories
def create_directories():
    dataset_path = "./face_dataset/"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    return dataset_path

create_directories()  # Ensure directory exists at startup

# Initialize session state
if "summarizer" not in st.session_state:
    st.session_state.summarizer = None
if "face_cascade" not in st.session_state:
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    st.session_state.face_cascade = cv2.CascadeClassifier(cascade_path)

# KNN for face recognition
def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
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

# Helper function to convert NumPy types for JSON serialization
def convert(obj):
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    return obj

# Main navigation using tabs instead of sidebar
tabs = st.tabs(["Home", "Face Recognition", "Face Analysis", "Text Summarization"])

# Home tab
with tabs[0]:
    st.markdown(
        "<h2 class='sub-header'>Welcome to the VisionText AI Hub!</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    This application offers three main functionalities:
    
    1. **Face Recognition**: Register new faces and recognize existing ones
    2. **Face Analysis**: Analyze faces for age, gender, emotion, and ethnicity
    3. **Text Summarization**: Generate concise summaries of longer texts
    
    Select a tab above to get started!
    """
    )

    # Display images for each functionality
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(
            "https://cdn.prod.website-files.com/66cd2cbca244787c4d744cc5/67362473a7b512c05960969b_OMY%20(1).jpeg",
            caption="Face Recognition",
        )
    with col2:
        st.image(
            "data:image/jpeg;base64,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",
            caption="Face Analysis",
        )
    with col3:
        st.image(
            "https://miro.medium.com/v2/resize:fit:1064/1*GIVviyN9Q0cqObcy-q-juQ.png",
            caption="Text Summarization",
        )

# Face Recognition tab
with tabs[1]:
    st.markdown("<h2 class='sub-header'>Face Recognition</h2>", unsafe_allow_html=True)

    face_recognition_mode = st.radio(
        "Choose an operation", ["Register New Face", "Recognize Faces"]
    )

    # Register New Face Section with WebRTC
    if face_recognition_mode == "Register New Face":
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("Register a New Face")

        file_name = st.text_input("Enter name for the face data:")
        st.info("Click 'Start' to begin the webcam. The app will collect face samples automatically.")
        
        class FaceRegisterProcessor(VideoProcessorBase):
            def __init__(self):
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.face_data = []
                self.counter = 0
                self.max_samples = 30
                self.name = ""
                
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.3, 
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # Only collect if name is set and we haven't reached max samples
                if self.name and len(self.face_data) < self.max_samples:
                    if len(faces) > 0:
                        # Sort faces by area (to pick the largest face)
                        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                        
                        # Take the largest face
                        for face in faces[:1]:
                            x, y, w, h = face
                            # Draw rectangle
                            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            
                            # Extract face section with padding
                            offset = 10
                            try:
                                face_section = img[
                                    max(0, y - offset) : min(img.shape[0], y + h + offset), 
                                    max(0, x - offset) : min(img.shape[1], x + w + offset)
                                ]
                                face_section = cv2.resize(face_section, (100, 100))
                                
                                # Only collect every few frames
                                self.counter += 1
                                if self.counter % 5 == 0:  # Collect every 5th frame
                                    self.face_data.append(face_section)
                                    
                                # Display sample count
                                cv2.putText(
                                    img,
                                    f"Samples: {len(self.face_data)}/{self.max_samples}",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 255),
                                    2,
                                )
                            except Exception as e:
                                pass
                else:
                    # Just display faces
                    for face in faces:
                        x, y, w, h = face
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            def set_name(self, name):
                self.name = name
            
            def get_face_data(self):
                if self.face_data:
                    return np.asarray(self.face_data)
                return None

        # Set up WebRTC streamer
        ctx = webrtc_streamer(key="face-register", video_processor_factory=FaceRegisterProcessor)
        
        # Status placeholder
        status = st.empty()
        
        # Start collecting data when button is pressed
        if ctx.video_processor and file_name:
            ctx.video_processor.set_name(file_name)
            
            # Save button
            if st.button("Save Face Data"):
                face_data = ctx.video_processor.get_face_data()
                
                if face_data is not None and len(face_data) > 0:
                    # Reshape and save
                    face_data = face_data.reshape((face_data.shape[0], -1))  # Flatten
                    dataset_path = create_directories()
                    np.save(os.path.join(dataset_path, file_name + ".npy"), face_data)
                    st.success(f"Face data saved for {file_name}! {len(face_data)} samples collected.")
                    
                    # Display a sample
                    if len(face_data) > 0:
                        sample = face_data[0].reshape(100, 100, 3).astype(np.uint8)
                        st.image(sample, caption=f"Sample for {file_name}", width=150)
                else:
                    st.error("No face data collected. Please try again.")
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Recognize Faces Section with WebRTC
    elif face_recognition_mode == "Recognize Faces":
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("Recognize Faces")
        st.info("Make sure you've registered faces first using the 'Register New Face' option.")
        
        # Load face data
        dataset_path = create_directories()
        face_data = []
        labels = []
        class_id = 0
        names = {}
        
        # Check if there's any face data available
        if not os.listdir(dataset_path):
            st.error("No face data found. Please register faces first.")
        else:
            for fx in os.listdir(dataset_path):
                if fx.endswith(".npy"):
                    names[class_id] = fx[:-4]
                    data_item = np.load(os.path.join(dataset_path, fx))
                    face_data.append(data_item)
                    target = class_id * np.ones((data_item.shape[0],))
                    labels.append(target)
                    class_id += 1
            
            if not face_data:
                st.error("No face data loaded. Please check the dataset.")
            else:
                face_dataset = np.concatenate(face_data, axis=0)
                face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
                trainset = np.concatenate((face_dataset, face_labels), axis=1)
                
                # Create WebRTC processor for face recognition
                class FaceRecognizeProcessor(VideoProcessorBase):
                    def __init__(self):
                        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                        self.names = names
                        self.trainset = trainset
                    
                    def recv(self, frame):
                        img = frame.to_ndarray(format="bgr24")
                        
                        # Convert to grayscale for face detection
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        
                        # Detect faces
                        faces = self.face_cascade.detectMultiScale(
                            gray, 
                            scaleFactor=1.3, 
                            minNeighbors=5,
                            minSize=(30, 30)
                        )
                        
                        for face in faces:
                            x, y, w, h = face
                            offset = 10
                            try:
                                face_section = img[
                                    max(0, y - offset) : min(img.shape[0], y + h + offset), 
                                    max(0, x - offset) : min(img.shape[1], x + w + offset)
                                ]
                                
                                if face_section.size == 0:
                                    continue
                                
                                face_section = cv2.resize(face_section, (100, 100))
                                # Flatten for prediction
                                out = knn(self.trainset, face_section.flatten())
                                name = self.names[int(out)]
                                
                                # Draw name and rectangle
                                cv2.putText(
                                    img,
                                    name,
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 255, 255),
                                    2,
                                    cv2.LINE_AA,
                                )
                                cv2.rectangle(
                                    img,
                                    (x, y),
                                    (x + w, y + h),
                                    (255, 255, 255),
                                    2,
                                )
                            except Exception as e:
                                # Just draw rectangle if recognition fails
                                cv2.rectangle(
                                    img,
                                    (x, y),
                                    (x + w, y + h),
                                    (0, 0, 255),
                                    2,
                                )
                        
                        return av.VideoFrame.from_ndarray(img, format="bgr24")
                
                # Run the webRTC streamer
                webrtc_streamer(key="face-recognize", video_processor_factory=FaceRecognizeProcessor)
                
                # Display info about registered faces
                with st.expander("View registered faces"):
                    st.write("People in the database:")
                    for name_id, name in names.items():
                        st.write(f"- {name}")
        
        st.markdown("</div>", unsafe_allow_html=True)

# Face Analysis tab
with tabs[2]:
    st.markdown("<h2 class='sub-header'>Face Analysis</h2>", unsafe_allow_html=True)
    st.write("Analyze faces for age, gender, emotion, and ethnicity.")
    
    # Helper function to safely convert NumPy types to standard Python types
    def convert(obj):
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        return obj
    
    # Function to analyze an image
    def analyze_image(img_path=None, img=None):
        try:
            with st.spinner("Analyzing face, please wait..."):
                # Configure DeepFace to avoid the initialization error
                backends = ['opencv', 'ssd', 'mtcnn', 'retinaface']
                detector_backend = backends[0]  # Start with opencv
                
                # Create a TensorFlow compatibility wrapper
                class TFCompatLayer(tf.keras.layers.Layer):
                    def __init__(self):
                        super(TFCompatLayer, self).__init__()
                        
                    def call(self, inputs):
                        # This wrapper ensures TensorFlow operations work with Keras tensors
                        return inputs
                
                # Set TensorFlow to use the compatibility mode
                tf.keras.backend.clear_session()
                
                # Try different backends if one fails
                for backend in backends:
                    try:
                        if img_path:
                            result = DeepFace.analyze(
                                img_path=img_path, 
                                actions=['age', 'gender', 'emotion', 'race'],
                                detector_backend=backend,
                                enforce_detection=False
                            )
                        else:
                            # Save the image temporarily to analyze it
                            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                                temp_filename = temp_file.name
                                cv2.imwrite(temp_filename, img)
                            
                            result = DeepFace.analyze(
                                img_path=temp_filename, 
                                actions=['age', 'gender', 'emotion', 'race'],
                                detector_backend=backend,
                                enforce_detection=False
                            )
                            # Clean up the temporary file
                            os.unlink(temp_filename)
                        
                        # If we get here, the analysis worked
                        detector_backend = backend
                        break
                    except Exception as e:
                        st.warning(f"Backend {backend} failed. Trying another...")
                        if backend == backends[-1]:  # If this was the last backend
                            raise e
                
                if isinstance(result, list):
                    result = result[0]
                
                analyzed_info = {
                    "Age": convert(result.get('age')),
                    "Gender": {k: convert(v) for k, v in result.get('gender', {}).items()},
                    "Emotion": result.get('emotion'),
                    "Race": result.get('race')
                }
                
                return analyzed_info
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            return None
    
    # Function to display analysis results
    def display_results(results):
        if not results:
            return
        
        # Display age
        st.subheader("Age")
        st.info(f"{results['Age']:.1f} years")
        
        # Display gender
        st.subheader("Gender")
        gender_df = pd.DataFrame({
            'Gender': list(results['Gender'].keys()),
            'Confidence': list(results['Gender'].values())
        })
        gender_df = gender_df.sort_values(by='Confidence', ascending=False)
        st.bar_chart(gender_df.set_index('Gender'))
        
        # Display emotion
        st.subheader("Emotion")
        emotion_df = pd.DataFrame({
            'Emotion': list(results['Emotion'].keys()),
            'Confidence': list(results['Emotion'].values())
        })
        emotion_df = emotion_df.sort_values(by='Confidence', ascending=False)
        st.bar_chart(emotion_df.set_index('Emotion'))
        
        # Display race/ethnicity
        st.subheader("Ethnicity")
        race_df = pd.DataFrame({
            'Ethnicity': list(results['Race'].keys()),
            'Confidence': list(results['Race'].values())
        })
        race_df = race_df.sort_values(by='Confidence', ascending=False)
        st.bar_chart(race_df.set_index('Ethnicity'))
    
    # Create tabs for different input methods within the face analysis tab
    face_tabs = st.tabs(["Upload Image", "Live Webcam", "Find Similar Faces"])
    
    with face_tabs[0]:
        st.header("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="face_uploader")
        
        if uploaded_file is not None:
            # Create two columns for image and results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Uploaded Image")
                # Read and display the image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                uploaded_file.seek(0)  # Reset file pointer
                image = cv2.imdecode(file_bytes, 1)
                st.image(image, channels="BGR")
                
                # Analysis button
                if st.button("Analyze Image", key="analyze_face_btn"):
                    # Save the uploaded file to a temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                        temp_filename = temp_file.name
                        temp_file.write(uploaded_file.getvalue())
                    
                    with col2:
                        st.subheader("Analysis Results")
                        results = analyze_image(img_path=temp_filename)
                        if results:
                            display_results(results)
                        
                        # Clean up the temporary file
                        os.unlink(temp_filename)
    
    with face_tabs[1]:
        st.header("Live Webcam Analysis")
        st.write("Click 'Start Camera' to begin")
        
        # Use session state to control the webcam flow
        if 'camera_running' not in st.session_state:
            st.session_state.camera_running = False
        
        # Button to toggle camera
        if st.session_state.camera_running:
            if st.button("Stop Camera", key="stop_face_cam"):
                st.session_state.camera_running = False
                st.rerun()
        else:
            if st.button("Start Camera", key="start_face_cam"):
                st.session_state.camera_running = True
                st.rerun()
        
        # Create placeholders for webcam feed and results
        webcam_placeholder = st.empty()
        results_placeholder = st.empty()
        
        # Handle the webcam
        if st.session_state.camera_running:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Failed to access webcam. Please make sure it's connected and not in use by another application.")
                st.session_state.camera_running = False
            else:
                # Capture button
                capture_clicked = st.button("Capture and Analyze", key="capture_face")
                
                # Show the webcam feed
                ret, frame = cap.read()
                if ret:
                    webcam_placeholder.image(frame, channels="BGR", caption="Live Webcam Feed")
                    
                    if capture_clicked:
                        # Use the captured frame for analysis
                        st.session_state.camera_running = False
                        webcam_placeholder.image(frame, channels="BGR", caption="Captured Image")
                        
                        # Analyze the captured image
                        with results_placeholder.container():
                            st.subheader("Analysis Results")
                            results = analyze_image(img=frame)
                            if results:
                                display_results(results)
                
                # Release the camera if we're done
                if not st.session_state.camera_running:
                    cap.release()
                    webcam_placeholder.empty()
                    results_placeholder.empty()
    
    with face_tabs[2]:
        st.header("Find Similar Faces")
        st.write("Upload an image to find similar faces in your dataset")
        
        # Setup for face database
        db_path = st.text_input("Enter path to your face database folder", value="./images/")
        model_name = st.selectbox("Select model", ["Facenet", "VGG-Face", "OpenFace", "DeepFace"], index=0)
        distance_metric = st.selectbox("Select distance metric", ["cosine", "euclidean", "euclidean_l2"], index=0)
        
        # Create dataset section
        with st.expander("Create/Manage Dataset"):
            if st.button("Create Database Directory"):
                os.makedirs(db_path, exist_ok=True)
                st.success(f"Created directory {db_path}")
            
            # Option to download LFW dataset
            if st.button("Download LFW Dataset"):
                with st.spinner("Downloading LFW dataset..."):
                    try:
                        from torchvision.datasets import LFWPeople
                        dataset = LFWPeople(root='./', download=True)
                        st.success("Downloaded LFW dataset successfully!")
                    except Exception as e:
                        st.error(f"Error downloading dataset: {str(e)}")
            
            # Option to sample from LFW to database
            if os.path.exists('./lfw-py/lfw_funneled/'):
                sampling_rate = st.slider("Sampling rate (% of images to copy)", min_value=1, max_value=50, value=20)
                
                if st.button("Sample Images to Database"):
                    with st.spinner("Sampling images..."):
                        os.makedirs(db_path, exist_ok=True)
                        sampled_count = 0
                        
                        for img in glob.glob('./lfw-py/lfw_funneled/*/*jpg', recursive=True):
                            if random.uniform(0,1) < (sampling_rate/100):
                                shutil.copy(img, db_path)
                                sampled_count += 1
                        
                        st.success(f"Done! Copied {sampled_count} images to {db_path}")
        
        # Upload and find similar faces
        uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"], key="similar_face_uploader")
        
        if uploaded_file is not None:
            # Display the uploaded image
            st.subheader("Query Image")
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            uploaded_file.seek(0)
            image = cv2.imdecode(file_bytes, 1)
            st.image(image, channels="BGR")
            
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_filename = temp_file.name
                temp_file.write(uploaded_file.getvalue())
            
            # Find similar faces button
            if st.button("Find Similar Faces"):
                try:
                    with st.spinner("Finding similar faces..."):
                        # Check if the database directory exists and has images
                        if not os.path.exists(db_path):
                            st.error(f"Database path {db_path} does not exist!")
                        else:
                            image_files = glob.glob(os.path.join(db_path, "*.jpg")) + \
                                         glob.glob(os.path.join(db_path, "*.jpeg")) + \
                                         glob.glob(os.path.join(db_path, "*.png"))
                            
                            if len(image_files) == 0:
                                st.warning(f"No images found in {db_path}. Please add some images first.")
                            else:
                                # Find similar faces
                                dfs = DeepFace.find(
                                    img_path=temp_filename,
                                    db_path=db_path,
                                    model_name=model_name,
                                    distance_metric=distance_metric,
                                    enforce_detection=False,
                                    silent=True
                                )
                                
                                if len(dfs) > 0 and len(dfs[0]) > 0:
                                    # Display results
                                    st.subheader("Similar Faces")
                                    
                                    # Limit to top 4 results
                                    num_results = min(4, len(dfs[0]))
                                    
                                    # Create a grid to display results
                                    cols = st.columns(num_results)
                                    
                                    for i in range(num_results):
                                        with cols[i]:
                                            img_path = dfs[0]['identity'][i]
                                            distance = dfs[0]['distance'][i]
                                            
                                            # Display the image
                                            img = plt.imread(img_path)
                                            st.image(img, caption=f"Similarity: {(1-distance)*100:.1f}%")
                                            
                                            # Analyze and display face attributes
                                            try:
                                                analysis = DeepFace.analyze(
                                                    img_path=img_path,
                                                    actions=['age', 'gender', 'emotion', 'race'],
                                                    enforce_detection=False,
                                                    silent=True
                                                )
                                                
                                                if isinstance(analysis, list):
                                                    analysis = analysis[0]
                                                
                                                st.write(f"Age: {analysis['age']:.1f}")
                                                st.write(f"Gender: {max(analysis['gender'], key=analysis['gender'].get)}")
                                                st.write(f"Emotion: {max(analysis['emotion'], key=analysis['emotion'].get)}")
                                                st.write(f"Ethnicity: {max(analysis['race'], key=analysis['race'].get)}")
                                            except Exception as e:
                                                st.error(f"Error analyzing similar face: {str(e)}")
                                else:
                                    st.warning("No similar faces found!")
                
                except Exception as e:
                    st.error(f"Error finding similar faces: {str(e)}")
                    
                # Clean up
                os.unlink(temp_filename)
                                
            # Advanced visualization option
            with st.expander("Advanced Visualization"):
                if st.button("Generate Visualization Panel"):
                    try:
                        with st.spinner("Generating visualization..."):
                            # Find similar faces
                            dfs = DeepFace.find(
                                img_path=temp_filename,
                                db_path=db_path,
                                model_name=model_name,
                                distance_metric=distance_metric,
                                enforce_detection=False,
                                silent=True
                            )
                            
                            if len(dfs) > 0 and len(dfs[0]) >= 3:
                                # Create visualization panel
                                panel = [plt.imread(temp_filename)]
                                
                                # Function to create text analysis image
                                def txt_analyze(d):
                                    # Create an image with the analysis results
                                    image = Image.new('RGB', (250, 100), 'white')
                                    draw = ImageDraw.Draw(image)
                                    
                                    # Try to use a system font
                                    try:
                                        font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf', 24)
                                    except:
                                        # Fallback to default
                                        font = ImageFont.load_default()
                                    
                                    gender = d['dominant_gender'] if 'dominant_gender' in d else max(d['gender'], key=d['gender'].get)
                                    age = d['age']
                                    emotion = d['dominant_emotion'] if 'dominant_emotion' in d else max(d['emotion'], key=d['emotion'].get)
                                    race = d['dominant_race'] if 'dominant_race' in d else max(d['race'], key=d['race'].get)
                                    
                                    draw.text((30, 5), f"{gender.lower()}, {age}", font=font, fill='blue')
                                    draw.text((30, 35), emotion, font=font, fill='blue')
                                    draw.text((30, 65), race, font=font, fill='blue')
                                    return np.array(image)
                                
                                # Analyze original image
                                query_obj = DeepFace.analyze(
                                    img_path=temp_filename,
                                    actions=['age', 'gender', 'emotion', 'race'],
                                    enforce_detection=False,
                                    silent=True
                                )
                                if isinstance(query_obj, list):
                                    query_obj = query_obj[0]
                                
                                # Add query image with analysis
                                panel[0] = np.vstack((panel[0], txt_analyze(query_obj)))
                                
                                # Add top 3 similar faces with analysis
                                for i in range(3):
                                    imgfile = dfs[0]['identity'][i]
                                    objs = DeepFace.analyze(
                                        img_path=imgfile,
                                        actions=['age', 'gender', 'emotion', 'race'],
                                        enforce_detection=False,
                                        silent=True
                                    )
                                    
                                    if isinstance(objs, list):
                                        objs = objs[0]
                                    
                                    panel += [np.vstack((plt.imread(imgfile), txt_analyze(objs)))]
                                
                                # Use matplotlib to create visualization
                                fig = plt.figure(figsize=(10., 10.))
                                grid = ImageGrid(fig, 111, nrows_ncols=(1, 4), axes_pad=0.1)
                                
                                for ax, im in zip(grid, panel):
                                    ax.set_axis_off()
                                    ax.imshow(im)
                                
                                # Display in Streamlit
                                st.pyplot(fig)
                            else:
                                st.warning("Not enough similar faces found for visualization!")
                    
                    except Exception as e:
                        st.error(f"Error generating visualization: {str(e)}")
                        
                    # Clean up
                    os.unlink(temp_filename)
                                        
# Text Summarization tab
with tabs[3]:
    st.markdown("<h2 class='sub-header'>Text Summarization</h2>", unsafe_allow_html=True)
    st.write("Generate concise summaries of longer texts.")

    st.markdown("<div class='section'>", unsafe_allow_html=True)

    # Model selection
    model_choice = st.selectbox(
        "Choose Model", 
        ["facebook/bart-large-cnn", "t5-small"],
        help="Select a model for text summarization"
    )
    
    # Initialize or update the summarizer when model changes
    @st.cache_resource
    def load_summarizer(model_name):
        return pipeline("summarization", model=model_name)
    
    try:
        st.session_state.summarizer = load_summarizer(model_choice)
        
        # Text input
        st.subheader("Enter text to summarize")
        text_to_summarize = st.text_area(
            "Paste your text here:", 
            height=200,
            placeholder="Paste your article, document, or any text you want to summarize here..."
        )
        
        # Summarization parameters
        st.subheader("Summarization Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            max_length = st.slider("Maximum summary length", 30, 500, 130)
        with col2:
            min_length = st.slider("Minimum summary length", 10, 100, 30)
        
        # Summarize button
        if st.button("Summarize") and text_to_summarize:
            if len(text_to_summarize) < 50:
                st.error("Please enter more text for a meaningful summary (at least 50 characters).")
            else:
                with st.spinner("Generating summary..."):
                    try:
                        summary = st.session_state.summarizer(
                            text_to_summarize,
                            max_length=max_length,
                            min_length=min_length,
                            do_sample=False,
                        )
                        
                        # Display the summary
                        st.subheader("Generated Summary")
                        st.info(summary[0]["summary_text"])
                        
                        # Display statistics
                        st.subheader("Statistics")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Original Length", len(text_to_summarize.split()))
                        with col2:
                            st.metric("Summary Length", len(summary[0]["summary_text"].split()))
                        
                        # Download option
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"summary_{timestamp}.txt"
                        with open(filename, "w") as f:
                            f.write(f"ORIGINAL TEXT:\n\n{text_to_summarize}\n\n")
                            f.write(f"SUMMARY:\n\n{summary[0]['summary_text']}")
                        
                        with open(filename, "r") as f:
                            st.download_button(
                                label="Download Summary",
                                data=f,
                                file_name=filename,
                                mime="text/plain",
                            )
                    
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
    except Exception as e:
        st.error(f"Failed to load summarization model: {str(e)}")
        st.info("Make sure you have installed the required packages: pip install transformers torch")

    # Additional info about text summarization
    with st.expander("ℹ️ About Text Summarization Models"):
        st.markdown("""
        - **facebook/bart-large-cnn**: A larger model with better quality but slower processing
        - **t5-small**: A smaller, faster model with good quality for shorter texts
        
        The summarization process uses natural language processing to identify key information 
        and create concise summaries while preserving the most important content.
        """)

    st.markdown("</div>", unsafe_allow_html=True)
