import cv2
import tkinter as tk
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque, Counter

# Load AI Model
model = tf.keras.models.load_model('asl_model.keras')
classes = np.load('classes.npy')

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# UI Stringing Variables
sentence = ""
last_prediction = ""
frames_held = 0
REQUIRED_FRAMES = 8
cooldown = 0
pred_buffer = deque(maxlen=7)

# Initialize Main GUI Window
root = tk.Tk()
root.title("Public Service Desk: Sign Language Translator")
root.geometry("900x750")
root.configure(bg="#2c3e50")

# Title Label
title_label = tk.Label(root, text="Live Sign Translation", font=("Helvetica", 24, "bold"), bg="#2c3e50", fg="white")
title_label.pack(pady=10)

# Live Detection Status Label
status_label = tk.Label(root, text="Hand Status: Waiting...", font=("Helvetica", 14), bg="#2c3e50", fg="#f1c40f")
status_label.pack(pady=2)

# Video Feed Label
video_label = tk.Label(root, bg="black")
video_label.pack(pady=10)

# Translated Text Display
text_box = tk.Label(root, text="Waiting for input...", font=("Helvetica", 20), bg="#ecf0f1", fg="#2c3e50", width=55, height=3, relief="solid")
text_box.pack(pady=15)

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

def update_frame():
    global sentence, last_prediction, frames_held, cooldown
    
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(frame_rgb)
        
        current_sign = "None"
        confidence_val = 0.0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                wrist = hand_landmarks.landmark[0]
                wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z
                raw_coords = []
                for lm in hand_landmarks.landmark:
                    raw_coords.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
                
                coords = np.array(raw_coords, dtype=np.float32).reshape(21, 3)
                max_dist = np.max(np.linalg.norm(coords, axis=1))
                if max_dist > 0:
                    coords = coords / max_dist
                
                prediction = model.predict(coords.reshape(1, 63), verbose=0)
                predicted_class_index = np.argmax(prediction[0])
                raw_sign = classes[predicted_class_index]
                confidence_val = prediction[0][predicted_class_index]
                current_sign = raw_sign

                if confidence_val > 0.50:
                    pred_buffer.append(raw_sign)

                if len(pred_buffer) > 0:
                    most_common_sign, _ = Counter(pred_buffer).most_common(1)[0]
                    
                    if cooldown > 0:
                        cooldown -= 1
                    else:
                        if most_common_sign == last_prediction:
                            frames_held += 1
                        else:
                            last_prediction = most_common_sign
                            frames_held = 1

                        if frames_held >= REQUIRED_FRAMES:
                            pred_lower = most_common_sign.lower()
                            if pred_lower == 'space':
                                sentence += " "
                            elif pred_lower in ('del', 'delete'):
                                sentence = sentence[:-1]
                            elif pred_lower != 'nothing':
                                sentence += most_common_sign

                            frames_held = 0
                            pred_buffer.clear()
                            cooldown = 12

                            text_box.config(text=sentence if sentence else "[Empty]")

            status_label.config(text=f"Detected: {current_sign} ({confidence_val * 100:.0f}%) | Hold: {frames_held}/{REQUIRED_FRAMES}")
        else:
            pred_buffer.clear()
            frames_held = 0
            last_prediction = ""
            status_label.config(text="Hand Status: No Hand Detected")

        # Convert frame to PIL image format for Tkinter UI
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        
    video_label.after(15, update_frame)

def clear_text():
    global sentence
    sentence = ""
    text_box.config(text="Waiting for input...")

def backspace_text():
    global sentence
    sentence = sentence[:-1]
    text_box.config(text=sentence if sentence else "Waiting for input...")

# Control Buttons Frame
btn_frame = tk.Frame(root, bg="#2c3e50")
btn_frame.pack(pady=5)

clear_btn = tk.Button(btn_frame, text="Clear All", font=("Helvetica", 12, "bold"), bg="#e74c3c", fg="white", command=clear_text)
clear_btn.pack(side="left", padx=5)

back_btn = tk.Button(btn_frame, text="Backspace", font=("Helvetica", 12, "bold"), bg="#f39c12", fg="white", command=backspace_text)
back_btn.pack(side="left", padx=5)

update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()