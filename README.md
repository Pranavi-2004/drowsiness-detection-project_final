Ah, got it! You want the **actual `requirements.txt` content** to be shown **outside the code blocks** in the README so readers can see it clearly, and all bash/code instructions stay clean. Here’s the cleaned-up, ready-to-paste `README.md`:

````markdown
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

Make sure your Python version is **3.11** for TensorFlow 2.14 compatibility. Then run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4️⃣ Run the project

If the main Python file is `main.py` (or `app.py`), run:

```bash
python main.py
```

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
* Use Python 3.11 for compatibility with TensorFlow 2.14
* Adjust the alert threshold in the code if needed

```

---


Do you want me to do that?
```
