import streamlit as st
import pyrebase
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import tempfile
import matplotlib.pyplot as plt
from collections import Counter
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Firebase configuration
firebaseConfig = {
    "apiKey": os.getenv("apiKey"),
    "authDomain": os.getenv("authDomain"),
    "databaseURL": os.getenv("databaseURL"),
    "projectId": os.getenv("projectId"),
    "storageBucket": os.getenv("storageBucket"),
    "messagingSenderId": os.getenv("messagingSenderId"),
    "appId": os.getenv("appId")
}

# Initialize Firebase
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()
db = firebase.database()
storage = firebase.storage()

# Load YOLO model
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load COCO class names
with open("yolo/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

def load_image(image_file):
    img = Image.open(image_file)
    return img

def detect_objects(img):
    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    detected_objects = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                detected_objects.append(classes[class_id])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)  # Green color for bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    
    return img, detected_objects

# Function to handle Firebase Authentication
def handle_user(auth_type):
    if auth_type == "Log In":
        st.subheader(":red[Log in]")

        email = st.text_input("Email", key="signin_email")
        password = st.text_input("Password", type="password", key="signin_password")

        signin_clicked = st.button("Log in", type="primary")

        if signin_clicked:
            try:
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state['authenticated'] = True
                st.session_state['user_info'] = user
                st.session_state['login_error'] = None
                st.success("Logged in successfully!")
                st.experimental_rerun()
            except Exception as e:
                st.session_state['login_error'] = "Invalid credentials or error occurred. Please try again."

        if 'login_error' in st.session_state and st.session_state['login_error']:
            st.error(st.session_state['login_error'])
    
    elif auth_type == "Sign Up":
        st.subheader(":red[Create a new account]")

        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")

        signup_clicked = st.button("Sign Up", type="primary")

        if signup_clicked:
            try:
                user = auth.create_user_with_email_and_password(email, password)
                st.success("Account created successfully! Please login.")
            except:
                st.error("Error creating account. Please try again.")

# Function to save history to Firebase Realtime Database
def save_history(user_id, detected_objects, image_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    image_url = upload_image(image_path, user_id, timestamp)
    history_data = {
        "timestamp": timestamp,
        "detected_objects": detected_objects,
        "image_url": image_url
    }
    db.child("users").child(user_id).child("history").push(history_data)

# Function to upload image to Firebase Storage
def upload_image(image_path, user_id, timestamp):
    storage_path = f"{user_id}/{timestamp}.png"
    storage.child(storage_path).put(image_path)
    image_url = storage.child(storage_path).get_url(None)
    return image_url

# Function to fetch history from Firebase Realtime Database
def fetch_history(user_id):
    history = db.child("users").child(user_id).child("history").get()
    return history.val() if history.each() else []

# Function to clear history from Firebase Realtime Database
def clear_history(user_id):
    db.child("users").child(user_id).child("history").remove()

# Function to generate a bar chart for detected objects
def generate_bar_chart(user_id):
    user_history = fetch_history(user_id)
    if user_history:
        detected_objects_list = []
        for entry in user_history:
            detected_objects_list.extend(user_history[entry].get('detected_objects', []))
        
        if detected_objects_list:
            object_counts = Counter(detected_objects_list)
            objects = list(object_counts.keys())
            counts = list(object_counts.values())

            # Plotting the bar chart
            fig, ax = plt.subplots()
            ax.bar(objects, counts, color='skyblue')
            ax.set_xlabel('Objects')
            ax.set_ylabel('Count')
            ax.set_title('Frequency of Detected Objects')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)
        else:
            st.write("No objects detected yet.")
    else:
        st.write("No history available.")

# Main logic with session
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False

if 'show_graph' not in st.session_state:
    st.session_state['show_graph'] = False

if st.session_state['authenticated']:
    user_id = st.session_state['user_info']['localId']
    st.sidebar.write(f"You are logged in.")
    st.sidebar.subheader(f"Welcome, {st.session_state['user_info']['email']}")
    if st.sidebar.button("Logout", type="primary"):
        st.session_state['authenticated'] = False
        st.experimental_rerun()

    st.sidebar.divider()
    st.sidebar.header("View Detected Object Frequency")
    # Add a button to toggle the graph
    if st.sidebar.button("Show/Hide Graph", type="primary"):
        st.session_state['show_graph'] = not st.session_state['show_graph']

    # Modified section to fetch and display history
    st.sidebar.divider()
    st.sidebar.header("Your History")
    user_history = fetch_history(user_id)
    # Add a button to clear history
    if st.sidebar.button("Clear History", type="primary"):
        clear_history(user_id)
        st.experimental_rerun()

    if user_history:
        # Sort history by timestamp in descending order (newest on top)
        sorted_history = dict(sorted(user_history.items(), key=lambda item: item[1]['timestamp'], reverse=True))
        
        for entry in sorted_history:
            st.sidebar.write(f"Timestamp: {sorted_history[entry]['timestamp']}")
            st.sidebar.image(sorted_history[entry]['image_url'], width=150)
            st.sidebar.write("Detected objects:")
            if 'detected_objects' in sorted_history[entry]:
                for obj in sorted_history[entry]['detected_objects']:
                    st.sidebar.write(f"- {obj}")
            st.sidebar.write("---")
    else:
        st.sidebar.write("No history available.")

    st.title("Welcome to Identif:red[AI]er")
    st.subheader("Transform Your Images into Intelligent Insights")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = load_image(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Detecting objects...")

        # Convert image to OpenCV format
        img_array = np.array(image)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Detect objects
        detected_img, detected_objects = detect_objects(img)

        # Convert back to PIL format
        detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
        detected_img = Image.fromarray(detected_img)

        st.image(detected_img, caption='Processed Image.', use_column_width=True)

        # Display detected objects
        st.write("Detected objects:")
        for obj in detected_objects:
            st.write(f"- {obj}")

        # Save image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            detected_img.save(tmp_file.name)
            tmp_file_path = tmp_file.name

        # Save history
        save_history(user_id, detected_objects, tmp_file_path)
    
    if st.session_state['show_graph']:
        st.subheader("Detected Objects Frequency")
        generate_bar_chart(user_id)
        if st.button("Close Graph"):
            st.session_state['show_graph'] = False
            st.experimental_rerun()
else:
    st.title("Unleash the Power of AI with Identif:red[AI]er")
    st.subheader("Your AI-Powered Object Detection Companion")
    st.write("Discover the power of artificial intelligence with IdentifAIer! Our state-of-the-art object detection app harnesses the latest in AI technology to identify objects in your images with unparalleled accuracy and speed. Whether you're a tech enthusiast, a professional, or just curious, IdentifAIer offers a seamless and intuitive experience to explore the capabilities of AI.")
    st.write("Sign in now to start detecting and naming objects with just a click! If you’re new here, join us and see what AI can do for you. Let's get started!")
    
    auth_type = st.selectbox("Choose Authentication Method", ["Log In", "Sign Up"])
    handle_user(auth_type)

footer = """
<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by <a style='display: block; text-align: center;' href="mailto:itbin-2110-0074@horizoncampus.edu.lk" target="_blank">Abisha Navaneethamani</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
