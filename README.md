


# Drowsiness Detection Using Deep Learning

## Introduction
This project implements a **Drowsiness Detection System** using Deep Learning.  
It is designed to detect when a person is getting drowsy or sleepy while driving, helping prevent accidents caused by fatigue.  
The system uses a trained neural network to monitor the eyes of a person through a webcam and gives an alert when drowsiness is detected.

This project is beginner-friendly and can be run on your local machine with minimal setup.

---

## Step-by-Step Instructions to Run Locally

### 1️⃣ Clone the repository
Open your terminal or command prompt and run:

```bash
git clone https://github.com/Pranavi-2004/drowsiness-detection-project_final.git
cd drowsiness-detection-project_final
````

### 2️⃣ Create a virtual environment (recommended)

This keeps project dependencies separate from your global Python setup:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies

Make sure your Python version is **3.11** for TensorFlow compatibility. Then run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Download Haar cascades for eye/face detection (if not included)

OpenCV uses XML files for detection. You can download them from OpenCV GitHub:

```bash
# Example for eye cascade
# Download and place in the project folder or utils/
wget https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_eye.xml
wget https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml
```

### 5️⃣ Check the model path

Make sure your pre-trained model (`.h5` file) is in the `models/` folder and your code points to it correctly:

```python
# Example in main.py
model = load_model("models/drowsiness_model.h5")
```

### 6️⃣ Ensure alert sound exists (if used)

If your code plays a sound for drowsiness detection, make sure the file exists:

```
assets/alert.wav
```

### 7️⃣ Run the project

```bash
python main.py
```

A webcam window should open, and the system will monitor your eyes. If drowsiness is detected, you will get an alert.

---

## Requirements

The project uses the following Python libraries:

Pillow
tensorflow==2.20.0
protobuf==3.20.\*
streamlit==1.5.1
opencv-python
click==7.1.2
pygame
altair==4

> Save these in `requirements.txt` for pip to install automatically.

---

## Project File Structure

```
drowsiness-detection-project_final/
│
├── main.py               # Main Python script to run the detection
├── requirements.txt      # Project dependencies
├── models/               # Pre-trained deep learning model files (.h5)
│   └── drowsiness_model.h5
├── utils/                # Helper functions for image preprocessing & prediction
│   └── utils.py
├── assets/               # Any images, sounds, or media files
│   └── alert.wav
├── haarcascades/         # Optional: downloaded OpenCV XML files
│   ├── haarcascade_eye.xml
│   └── haarcascade_frontalface_default.xml
└── README.md             # Project documentation
```

---

## Code Overview

**main.py or app.py**

* Loads the trained deep learning model (`.h5` file)
* Captures video from the webcam using OpenCV
* Detects eyes using OpenCV’s Haar cascades
* Predicts if eyes are **open** or **closed**
* Triggers an alert if drowsiness is detected for a continuous period

**utils/**

* Contains helper functions for preprocessing images, predicting eye states, etc.

---

## How It Works

1. Webcam captures video frames
2. Eyes are detected in each frame
3. The neural network predicts whether eyes are open or closed
4. If the eyes are closed for too long, an alert sound is triggered

---

## Notes

* Make sure your webcam is working
* Use Python 3.11 for compatibility with TensorFlow 2.14+
* Download Haar cascade XML files if not included
* Make sure your `.h5` model path is correct in the code
* Ensure alert sound file exists if used
* Adjust the alert threshold in the code if needed


