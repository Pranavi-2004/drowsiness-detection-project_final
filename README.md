# Drowsiness Detection Using Deep Learning

## Introduction
This project implements a **Drowsiness Detection System** using Deep Learning.  
It is designed to detect when a person is getting drowsy or sleepy while driving, helping prevent accidents caused by fatigue.  
The system uses a trained neural network to monitor the eyes of a person through a webcam and gives an alert when drowsiness is detected.

This project is beginner-friendly and can be run on your local machine with minimal setup.

---

## Step-by-Step Instructions to Run Locally

###  Clone the repository
Open your terminal or command prompt and run:

```bash
git clone https://github.com/Pranavi-2004/drowsiness-detection-project_final.git
cd drowsiness-detection-project_final

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Code review
# Loads the trained deep learning model (.h5 file)
# Captures video from the webcam using OpenCV
# Detects eyes using OpenCV’s Haar cascades
# Predicts if eyes are open or closed
# Triggers an alert if drowsiness is detected for a continuous period
