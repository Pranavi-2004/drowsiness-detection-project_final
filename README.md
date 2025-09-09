 Drowsiness Detection Using Deep Learning

Introduction
This project implements a **Drowsiness Detection System** using Deep Learning.
It is designed to detect when a person is getting drowsy or sleepy while driving, helping prevent accidents caused by fatigue.
The system uses a trained neural network to monitor the eyes of a person through a webcam and gives an alert when drowsiness is detected.

This project is beginner-friendly and can be run on your local machine with minimal setup.

Step-by-Step Instructions to Run Locally:
Follow these steps to get the project up and running:

Clone the repository
Open your terminal or command prompt and run:
git clone https://github.com/Pranavi-2004/drowsiness-detection-project_final.git
cd drowsiness-detection-project_final

Create a virtual environment (recommended)
This keeps project dependencies separate from your global Python setup:
# Windows
python -m venv venv
venv\Scripts\activate
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
Install dependencies

Make sure your Python version is 3.11 for TensorFlow 2.14 compatibility. Then install required libraries:
pip install --upgrade pip
pip install -r requirements.txt

Run the project
python main.py



Loads the trained deep learning model (.h5 file).
Captures video from the webcam using OpenCV.
Detects eyes using OpenCV’s Haar cascades.
Predicts if eyes are open or closed.
Triggers an alert if drowsiness is detected for a continuous period.

requirements.txt
Lists all Python libraries needed to run the project.

models/ folder
Contains the pre-trained model files for drowsiness detection.

utils
Helper functions for preprocessing images, predicting eye states, etc.


How It Works
Webcam captures video frames.
Eyes are detected in each frame.
The neural network predicts whether eyes are open or closed.
If the eyes are closed for too long, an alert sound is triggered.


Make sure your webcam is working.
Use Python 3.11 for compatibility with TensorFlow 2.14.
Adjust the alert threshold in the code if needed.
