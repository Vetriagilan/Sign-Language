import os
import cv2
import numpy as np
import mediapipe as mp

# Try importing tqdm for progress bar; fall back to standard print if unavailable
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

def extract_landmarks(image_path, hands):
    """
    Reads an image and extracts 21 3D hand landmarks.
    Normalizes coordinates relative to wrist and scales by maximum hand distance for scale invariance.
    Returns (features, features_flipped) or (None, None).
    """
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    # Convert BGR to RGB for MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        wrist = hand_landmarks.landmark[0]
        wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

        raw_coords = []
        for lm in hand_landmarks.landmark:
            raw_coords.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
            
        coords = np.array(raw_coords, dtype=np.float32).reshape(21, 3)

        # 1. Scale Normalization: divide by max distance from wrist
        distances = np.linalg.norm(coords, axis=1)
        max_dist = np.max(distances)
        if max_dist > 0:
            coords = coords / max_dist
            
        features = coords.flatten()

        # 2. Horizontal Mirroring Augmentation (flip X coordinate) for left/right hand support
        coords_flipped = coords.copy()
        coords_flipped[:, 0] = -coords_flipped[:, 0]
        features_flipped = coords_flipped.flatten()

        return features, features_flipped

    return None, None

def main():
    dataset_dir = os.path.join(os.path.dirname(__file__), "asl_alphabet_train", "asl_alphabet_train")
    output_features_file = os.path.join(os.path.dirname(__file__), "features.npy")
    output_labels_file = os.path.join(os.path.dirname(__file__), "labels.npy")

    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found.")
        return

    print("Initializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )

    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    classes.sort()
    print(f"Found {len(classes)} classes: {classes}")

    X = []
    y = []
    
    # Limit per class (None for full dataset)
    MAX_SAMPLES_PER_CLASS = 2000

    for class_name in classes:
        class_dir = os.path.join(dataset_dir, class_name)
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if MAX_SAMPLES_PER_CLASS:
            image_files = image_files[:MAX_SAMPLES_PER_CLASS]

        print(f"Processing class '{class_name}' ({len(image_files)} images)...")
        
        extracted_count = 0
        for img_name in tqdm(image_files, desc=f"Class {class_name}"):
            img_path = os.path.join(class_dir, img_name)
            features, features_flipped = extract_landmarks(img_path, hands)
            if features is not None:
                # Add original and mirrored features
                X.append(features)
                y.append(class_name)
                X.append(features_flipped)
                y.append(class_name)
                extracted_count += 1
                
        print(f"Successfully extracted {extracted_count}/{len(image_files)} hands from class '{class_name}'.")

    hands.close()

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    print(f"\nExtraction completed!")
    print(f"Total samples collected (with mirror augmentation): {len(X)}")
    print(f"Feature matrix shape: {X.shape}")

    np.save(output_features_file, X)
    np.save(output_labels_file, y)
    print(f"Saved features to '{output_features_file}' and labels to '{output_labels_file}'.")

if __name__ == "__main__":
    main()
