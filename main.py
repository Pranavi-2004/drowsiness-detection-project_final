import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer

# Initialize alarm sound
mixer.init()
sound = mixer.Sound(r'C:\Users\B Pranavi\Auproject\Driver-Drowsiness-Detection-using-Deep-Learning\alarm.wav')

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Load model silently
model_path = r"C:\Users\B Pranavi\Auproject\Driver-Drowsiness-Detection-using-Deep-Learning\models\model.keras"

model = load_model(model_path)



lbl = ['Close', 'Open']
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, minNeighbors=3, scaleFactor=1.1, minSize=(25, 25))
    eyes = eye_cascade.detectMultiScale(gray, minNeighbors=1, scaleFactor=1.1)

    cv2.rectangle(frame, (0, height-50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    for (x, y, w, h) in eyes:
        eye = frame[y:y + h, x:x + w]
        eye = cv2.resize(eye, (80, 80))
        eye = eye / 255
        eye = eye.reshape(80, 80, 3)
        eye = np.expand_dims(eye, axis=0)

        prediction = model.predict(eye, verbose=0)

        if prediction[0][0] > 0.30:
            cv2.putText(frame, "Closed", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            score += 1
            if score > 5:
                try:
                    sound.play()
                except:
                    pass
        elif prediction[0][1] > 0.70:
            score -= 1
            score = max(score, 0)
            cv2.putText(frame, "Open", (10, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, 'Score:' + str(score), (100, height-20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Driver Drowsiness Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
