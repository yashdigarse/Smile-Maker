import streamlit as st
import dlib
import cv2
import numpy as np
from PIL import Image

# Load pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def add_smile(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        # Get coordinates of mouth landmarks
        mouth_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 61)]
        mouth_points = np.array(mouth_points, dtype=np.int32)

        # Create a smile effect by moving the corners of the mouth upwards
        mouth_points[3][1] -= 10  # Left corner
        mouth_points[9][1] -= 10  # Right corner

        # Draw the modified mouth on the image
        cv2.polylines(image, [mouth_points], isClosed=True, color=(0, 255, 0), thickness=2)
    return image

st.title("Smile Maker")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption='Original Image', use_column_width=True)
    st.write("")
    st.write("Processing...")

    result_image = add_smile(image.copy())
    st.image(result_image, caption='Smiling Image', use_column_width=True)