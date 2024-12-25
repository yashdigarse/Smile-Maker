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
        for n in range(48, 61):  # Mouth landmarks
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        # Modify landmarks to create a smile effect (simplified example)
        landmarks.part(48).y += 10
        landmarks.part(54).y += 10
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