import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque, Counter

# 1. Load the AI Model and Class Labels
print("Loading AI Model...")
model = tf.keras.models.load_model('asl_model.keras')
classes = np.load('classes.npy')
print(f"Loaded {len(classes)} classes: {classes}")

# 2. Set up MediaPipe Hand Detector
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5
)

# 3. Sentence & Buffer Logic Variables
sentence = ""
last_prediction = ""
frames_held = 0
REQUIRED_FRAMES = 8  # Number of consistent frames to lock in a letter (~0.25s)
cooldown = 0
pred_buffer = deque(maxlen=7)

# 4. Fire up the Webcam
cap = cv2.VideoCapture(0)
print("Webcam live! Press 'q' to quit, 'c' to clear sentence.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    current_sign = "None"
    confidence_val = 0.0
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract and scale-normalize 63 landmark coordinates
            wrist = hand_landmarks.landmark[0]
            wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

            raw_coords = []
            for lm in hand_landmarks.landmark:
                raw_coords.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
            
            coords = np.array(raw_coords, dtype=np.float32).reshape(21, 3)
            distances = np.linalg.norm(coords, axis=1)
            max_dist = np.max(distances)
            if max_dist > 0:
                coords = coords / max_dist

            input_data = coords.reshape(1, 63)
            
            # Predict sign
            prediction = model.predict(input_data, verbose=0)
            predicted_class_index = np.argmax(prediction[0])
            raw_sign = classes[predicted_class_index]
            confidence_val = prediction[0][predicted_class_index]
            
            current_sign = raw_sign

            if confidence_val > 0.50:
                pred_buffer.append(raw_sign)
            
            # Smooth predictions using majority vote over rolling buffer
            if len(pred_buffer) > 0:
                most_common_sign, count = Counter(pred_buffer).most_common(1)[0]
                
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

                        # Lock-in completed: reset hold & add brief cooldown
                        frames_held = 0
                        pred_buffer.clear()
                        cooldown = 12

    else:
        pred_buffer.clear()
        frames_held = 0
        last_prediction = ""

    # UI Overlay
    cv2.putText(frame, f'Sentence: {sentence}', (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, f'Sign: {current_sign} ({confidence_val * 100:.0f}%) | Hold: {frames_held}/{REQUIRED_FRAMES}', 
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)

    cv2.imshow('Sign Language Translator', frame)
    
    # Press 'q' to exit, 'c' to clear sentence
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ""

cap.release()
cv2.destroyAllWindows()