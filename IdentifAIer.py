import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import tempfile
import matplotlib.pyplot as plt
from collections import Counter
from io import BytesIO

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

    class_ids, confidences, boxes, detected_objects = [], [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
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

def save_history(detected_objects, image):
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_data = buffered.getvalue()

    if not st.session_state.get('last_uploaded') == img_data:
        st.session_state['history'].append({
            'image': img_data,
            'detected_objects': detected_objects,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.session_state['last_uploaded'] = img_data

def fetch_history():
    return st.session_state.get('history', [])

def clear_history():
    st.session_state['history'] = []
    st.session_state['last_uploaded'] = None
    st.rerun()

def display_history():
    user_history = fetch_history()
    st.sidebar.header("Detection History")
    if user_history:
        for idx, record in enumerate(user_history):
            st.sidebar.write(f"Image {idx + 1} - {record['timestamp']}")
            img = Image.open(BytesIO(record['image']))
            st.sidebar.image(img, width=100, caption=f"Detected: {', '.join(record['detected_objects'])}")
            st.sidebar.write("---")
        if st.sidebar.button("Clear History"):
            clear_history()
    else:
        st.sidebar.write("No history available.")

def generate_bar_chart():
    history = fetch_history()
    if history:
        detected_objects_list = [obj for record in history for obj in record['detected_objects']]
        if detected_objects_list:
            object_counts = Counter(detected_objects_list)
            objects, counts = zip(*object_counts.items())

            fig, ax = plt.subplots()
            ax.bar(objects, counts, color='skyblue')
            ax.set_xlabel('Objects')
            ax.set_ylabel('Count')
            ax.set_title('Frequency of Detected Objects')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

st.title("Welcome to Identif:red[AI]er")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = load_image(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    img_array = np.array(image)
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    detected_img, detected_objects = detect_objects(img)
    detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
    detected_img = Image.fromarray(detected_img)

    st.image(detected_img, caption='Processed Image.', use_column_width=True)
    st.write("Detected objects:")
    for obj in detected_objects:
        st.write(f"- {obj}")

    save_history(detected_objects, detected_img)

display_history()

st.sidebar.header("View Detected Object Frequency")
if st.sidebar.button("Show/Hide Graph"):
    st.session_state['show_graph'] = not st.session_state.get('show_graph', False)
if st.session_state.get('show_graph', False):
    st.subheader("Detected Objects Frequency")
    generate_bar_chart()

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
