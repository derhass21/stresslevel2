import cv2
import numpy as np
import random
import streamlit as st

# Fungsi untuk mendeteksi wajah dan menambahkan level stres pada frame
def detect_face_with_stress_level(frame, stress_level):
    # Deteksi wajah dengan Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Gambar kotak di sekitar wajah dan tambahkan "Stress Level"
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        label = f"Stress Level: {stress_level}"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = x + (w - text_size[0]) // 2  # Pusatkan teks pada kotak
        text_y = y - 10  # Letakkan di atas kotak
        cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
    
    return frame

# Streamlit Web UI
st.title('Live Face Detection with Stress Level')

# Tentukan angka "Stress Level" tetap untuk sesi ini
stress_level = random.randint(52, 91)

# Mengambil video dari webcam dan melakukan deteksi wajah
video_capture = cv2.VideoCapture(0)

# Jika webcam berhasil dibuka
if not video_capture.isOpened():
    st.error("Could not open webcam.")
else:
    stframe = st.empty()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Lakukan deteksi wajah pada frame
        frame = detect_face_with_stress_level(frame, stress_level)
        
        # Mengonversi frame dari BGR ke RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Menampilkan frame menggunakan Streamlit
        stframe.image(frame_rgb)

# Menutup webcam
video_capture.release()