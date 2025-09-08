import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer

# === Paths ===
alarm_path = r"C:\Users\B Pranavi\Auproject\Driver-Drowsiness-Detection-using-Deep-Learning\alarm.wav"
model_path = r"C:\Users\B Pranavi\Auproject\Driver-Drowsiness-Detection-using-Deep-Learning\models\model.h5"
# If your model is .keras format, swap the above line with:
# model_path = r"C:\Users\B Pranavi\Auproject\Driver-Drowsiness-Detection-using-Deep-Learning\models\model.keras"

# === Check files ===
if not os.path.exists(alarm_path):
    raise FileNotFoundError(f"alarm.wav not found at: {alarm_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at: {model_path}")

# === Load sound ===
mixer.init()
sound = mixer.Sound(alarm_path)

# === Load Haar cascades ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# === Load model ===
model = load_model(model_path)

lbl = ['Close', 'Open']

# === Video capture (use DirectShow backend on Windows) ===
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("❌ Error opening video stream. Check your webcam permissions.")

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to grab frame, retrying...")
        continue

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, minNeighbors=3, scaleFactor=1.1, minSize=(25, 25))
    eyes = eye_cascade.detectMultiScale(gray, minNeighbors=1, scaleFactor=1.1)

    # Bottom rectangle for text
    cv2.rectangle(frame, (0, height - 50), (250, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for (x, y, w, h) in eyes:
        eye = frame[y:y + h, x:x + w]
        eye = cv2.resize(eye, (80, 80))
        eye = eye / 255.0
        eye = eye.reshape(80, 80, 3)
        eye = np.expand_dims(eye, axis=0)

        prediction = model.predict(eye, verbose=0)

        if prediction[0][0] > 0.30:  # Closed
            cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            score += 1
            if score > 20:
                try:
                    if not mixer.get_busy():  # play only if not already playing
                        sound.play()
                except Exception as e:
                    print(f"⚠️ Sound play error: {e}")
        elif prediction[0][1] > 0.70:  # Open
            score = max(score - 1, 0)
            cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, 'Score:' + str(score), (120, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Driver Drowsiness Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
