# IdentifAIer

## Overview
IdentifAIer is an AI-powered object detection web application built using Streamlit. This app uses a pre-trained YOLO model to identify objects in images, authenticate users via Firebase, and store detection history. Users can upload images, detect objects, and view the frequency of detected objects over time through a dynamic bar chart.

## Features:
1. User Authentication: Sign up and log in using Firebase Authentication.
2. Object Detection: Upload images and detect objects using the YOLO model.
3. History Management: Save and view the history of detected objects.
4. Data Visualization: Display a bar chart showing the frequency of detected objects.
5. Firebase Integration: Store history and images in Firebase Realtime Database and Firebase Storage.
   
## Requirements:
- Python 3.8+
- Streamlit
- Pyrebase
- OpenCV
- NumPy
- Pillow
- Matplotlib
- Python-dotenv