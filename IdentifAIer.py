import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import tempfile
import matplotlib.pyplot as plt
from collections import Counter

net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

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
            color = (0, 255, 0) 
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    
    return img, detected_objects

def save_history(detected_objects, image_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_entry = {
        "timestamp": timestamp,
        "detected_objects": detected_objects,
        "image_path": image_path
    }
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    
    st.session_state['history'].insert(0, history_entry) 
    
def fetch_history():
    return st.session_state.get('history', [])

def clear_history():
    st.session_state['history'] = []

def generate_bar_chart():
    user_history = fetch_history()
    if user_history:
        detected_objects_list = []
        for entry in user_history:
            detected_objects_list.extend(entry.get('detected_objects', []))
        
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
        
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = True 

if 'show_graph' not in st.session_state:
    st.session_state['show_graph'] = False

st.sidebar.header("View Detected Object Frequency")
if st.sidebar.button("Show/Hide Graph"):
    st.session_state['show_graph'] = not st.session_state['show_graph']

st.sidebar.header("Your History")
user_history = fetch_history()

if st.sidebar.button("Clear History"):
    clear_history()
    st.rerun()

if user_history:
    for entry in user_history:
        st.sidebar.write(f"Timestamp: {entry['timestamp']}")
        st.sidebar.image(entry['image_path'], width=150)
        st.sidebar.write("Detected objects:")
        for obj in entry['detected_objects']:
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
    st.write("Detecting objects...")

    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    detected_img, detected_objects = detect_objects(img)
    detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
    detected_img = Image.fromarray(detected_img)

    st.image(detected_img, caption='Processed Image.', use_column_width=True)
    st.write("Detected objects:")
    for obj in detected_objects:
        st.write(f"- {obj}")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
        detected_img.save(tmp_file.name)
        tmp_file_path = tmp_file.name
        
    save_history(detected_objects, tmp_file_path)

if st.session_state['show_graph']:
    st.subheader("Detected Objects Frequency")
    generate_bar_chart()
    if st.button("Close Graph"):
        st.session_state['show_graph'] = False
        st.rerun()

footer = """
<style>
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
