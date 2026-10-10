import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

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

# 3. Fire up the Webcam
cap = cv2.VideoCapture(0)
print("Webcam live! Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break
        
    # Flip the frame horizontally for a natural mirror view
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to find hand landmarks
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmark skeleton on video stream
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract 63 relative coordinates (normalized relative to wrist)
            wrist = hand_landmarks.landmark[0]
            wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

            raw_coords = []
            for lm in hand_landmarks.landmark:
                raw_coords.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
            
            coords = np.array(raw_coords, dtype=np.float32).reshape(21, 3)

            # Apply scale normalization (divide by max Euclidean distance from wrist)
            distances = np.linalg.norm(coords, axis=1)
            max_dist = np.max(distances)
            if max_dist > 0:
                coords = coords / max_dist

            # Format input for model shape (1, 63)
            input_data = coords.reshape(1, 63)
            
            # Predict sign using model
            prediction = model.predict(input_data, verbose=0)
            predicted_class_index = np.argmax(prediction[0])
            predicted_sign = classes[predicted_class_index]
            confidence = prediction[0][predicted_class_index]
            
            # Display prediction on video overlay
            if confidence > 0.50:
                cv2.putText(
                    frame, 
                    f'Translation: {predicted_sign} ({confidence * 100:.0f}%)', 
                    (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, 
                    (0, 255, 0), 
                    3, 
                    cv2.LINE_AA
                )

    # Show video window
    cv2.imshow('Sign Language Translator', frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()
