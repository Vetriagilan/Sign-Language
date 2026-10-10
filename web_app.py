import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter
from flask import Flask, render_template, request, jsonify, Response

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load Trained Model & Classes
print("Loading Model and Classes for Web App...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'asl_model.keras')
CLASSES_PATH = os.path.join(os.path.dirname(__file__), 'classes.npy')

model = None
classes = []

if os.path.exists(MODEL_PATH) and os.path.exists(CLASSES_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH)
    print(f"Model loaded successfully with {len(classes)} classes.")
else:
    print("Warning: Model or classes.npy not found!")

# MediaPipe Initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Global State for Web Server Video Stream
state = {
    'sentence': '',
    'current_sign': 'None',
    'confidence': 0.0,
    'frames_held': 0,
    'required_frames': 8,
    'cooldown': 0,
    'last_prediction': '',
    'probs': {}
}

pred_buffer = deque(maxlen=7)
cap = None

def generate_video_stream():
    global cap, state, pred_buffer
    
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

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

                if model is not None:
                    prediction = model.predict(coords.reshape(1, 63), verbose=0)
                    idx = np.argmax(prediction[0])
                    raw_sign = str(classes[idx])
                    confidence_val = float(prediction[0][idx])
                    current_sign = raw_sign

                    # Top probabilities
                    top_indices = np.argsort(prediction[0])[-3:][::-1]
                    state['probs'] = {str(classes[i]): float(prediction[0][i]) for i in top_indices}

                    if confidence_val > 0.50:
                        pred_buffer.append(raw_sign)

                    if len(pred_buffer) > 0:
                        most_common_sign, _ = Counter(pred_buffer).most_common(1)[0]
                        
                        if state['cooldown'] > 0:
                            state['cooldown'] -= 1
                        else:
                            if most_common_sign == state['last_prediction']:
                                state['frames_held'] += 1
                            else:
                                state['last_prediction'] = most_common_sign
                                state['frames_held'] = 1

                            if state['frames_held'] >= state['required_frames']:
                                pred_lower = most_common_sign.lower()
                                if pred_lower == 'space':
                                    state['sentence'] += " "
                                elif pred_lower in ('del', 'delete'):
                                    state['sentence'] = state['sentence'][:-1]
                                elif pred_lower != 'nothing':
                                    state['sentence'] += most_common_sign

                                state['frames_held'] = 0
                                pred_buffer.clear()
                                state['cooldown'] = 12

        else:
            pred_buffer.clear()
            state['frames_held'] = 0
            state['last_prediction'] = ""
            state['probs'] = {}

        state['current_sign'] = current_sign
        state['confidence'] = confidence_val

        # Draw Overlay Text on Video Frame
        cv2.putText(frame, f"Sentence: {state['sentence']}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Sign: {current_sign} ({confidence_val * 100:.0f}%) | Hold: {state['frames_held']}/{state['required_frames']}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2, cv2.LINE_AA)

        # Encode Frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', classes=classes.tolist() if len(classes) > 0 else [])

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.get_json()
    if not data or 'landmarks' not in data:
        return jsonify({'error': 'No landmarks provided'}), 400
    
    try:
        raw_coords = np.array(data['landmarks'], dtype=np.float32)
        if raw_coords.shape != (63,):
            return jsonify({'error': 'Invalid landmarks shape, expected 63 values'}), 400
        
        pts = raw_coords.reshape(21, 3)
        
        # Scale normalization (divide by max Euclidean distance from wrist)
        dists = np.linalg.norm(pts, axis=1)
        max_d = np.max(dists)
        if max_d > 0:
            pts = pts / max_d
            
        input_data = pts.reshape(1, 63)
        
        # Inference
        pred = model.predict(input_data, verbose=0)
        idx = np.argmax(pred[0])
        predicted_sign = str(classes[idx])
        confidence = float(pred[0][idx])
        
        top_indices = np.argsort(pred[0])[-3:][::-1]
        top_probs = {str(classes[i]): float(pred[0][i]) for i in top_indices}

        return jsonify({
            'sign': predicted_sign,
            'confidence': confidence,
            'probabilities': top_probs
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/state')
def get_state():
    return jsonify({
        'sentence': state['sentence'],
        'current_sign': state['current_sign'],
        'confidence': state['confidence'],
        'frames_held': state['frames_held'],
        'required_frames': state['required_frames'],
        'probs': state['probs']
    })

@app.route('/api/clear', methods=['POST'])
def clear_sentence():
    state['sentence'] = ""
    return jsonify({'status': 'cleared', 'sentence': ''})

@app.route('/api/backspace', methods=['POST'])
def backspace_sentence():
    state['sentence'] = state['sentence'][:-1]
    return jsonify({'status': 'backspaced', 'sentence': state['sentence']})

@app.route('/api/space', methods=['POST'])
def space_sentence():
    state['sentence'] += " "
    return jsonify({'status': 'space_added', 'sentence': state['sentence']})

if __name__ == '__main__':
    print("\n========================================================")
    print("Sign Language Web Translator Server Starting...")
    print("Open http://localhost:5000 in your browser")
    print("========================================================\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
