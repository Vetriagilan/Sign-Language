# Real-Time Sign Language Translator (ASL)

A computer vision and deep learning application for real-time American Sign Language (ASL) recognition and sentence translation using OpenCV, MediaPipe 3D Hand Landmarks, TensorFlow/Keras, and Tkinter.

---

## 🌟 Key Features

* **3D Hand Landmark Feature Extraction**: Extracts 21 3D hand landmarks using MediaPipe.
* **Scale & Distance Invariance**: Normalizes landmark vectors by Euclidean distance from the wrist, ensuring stable recognition regardless of hand position or distance to camera.
* **Left & Right Hand Recognition**: Trained with horizontal mirroring data augmentation to support both left and right hands seamlessly.
* **Rolling Prediction Smoothing**: Majority voting buffer filters out single-frame camera noise or hand jitter.
* **Desktop GUI App**: Interactive Tkinter application with live detection status, confidence metrics, and text controls.

---

## 📁 Repository Structure

```text
├── app.py                 # Main Tkinter Desktop Application
├── live_translation2.py   # OpenCV real-time translation with text stringing
├── live_translation.py    # OpenCV baseline real-time recognition
├── extract_features.py    # Feature extraction & scale normalization script
├── train_model.py         # Model training script
├── webcam_test.py         # MediaPipe webcam connectivity test
├── asl_model.keras        # Trained Keras Neural Network model
├── classes.npy            # Array of class labels (A-Z, space, del)
├── requirements.txt       # Python project dependencies
└── .gitignore             # Git ignore configuration
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install required packages:

```bash
git clone <your-repository-url>
cd Sign_language
pip install -r requirements.txt
```

### 2. Run the Applications

#### Option A: Interactive Desktop App (Tkinter GUI)
```bash
python app.py
```

#### Option B: Live Translation Overlay (OpenCV Window)
```bash
python live_translation2.py
```

---

## 🏋️ Model Training Pipeline

If you wish to re-extract features or retrain the neural network:

1. **Extract Features**:
   ```bash
   python extract_features.py
   ```
2. **Train Model**:
   ```bash
   python train_model.py
   ```

---

## 📋 License

This project is licensed under the MIT License.
