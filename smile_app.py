import streamlit as st
import cv2
import dlib
import numpy as np
from PIL import Image

# Load the pre-trained facial landmark detector
predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def add_smile(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        for i in range(48, 55):
            landmarks.part(i).y -= 5  # Move upper lip up
        for i in range(55, 61):
            landmarks.part(i).y += 5  # Move lower lip down
        for n in range(48, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    return image

st.title("Smile Effect App")
st.write("Upload an image and see the magic!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Applying smile effect...")
    result_image = add_smile(image)
    st.image(result_image, caption='Image with Smile Effect', use_column_width=True)