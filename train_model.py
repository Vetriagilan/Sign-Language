import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Load the structured data
print("Loading data...")
X_raw = np.load('features.npy')
y_raw = np.load('labels.npy')

# Ensure scale normalization (divide by max Euclidean distance from wrist)
X_norm = []
for row in X_raw:
    pts = row.reshape(21, 3)
    dists = np.linalg.norm(pts, axis=1)
    max_d = np.max(dists)
    if max_d > 0:
        pts = pts / max_d
    X_norm.append(pts.flatten())

X_norm = np.array(X_norm, dtype=np.float32)

# Generate horizontal mirroring data augmentation (flip X) if dataset doesn't already contain it
if len(X_norm) == len(y_raw):
    print("Applying horizontal mirroring data augmentation...")
    X_flip = X_norm.copy().reshape(-1, 21, 3)
    X_flip[:, :, 0] = -X_flip[:, :, 0]
    X_flip = X_flip.reshape(-1, 63)
    X_aug = np.vstack([X_norm, X_flip])
    y_aug = np.hstack([y_raw, y_raw])
else:
    X_aug = X_norm
    y_aug = y_raw

# 2. Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_aug)
num_classes = len(label_encoder.classes_)

# Save the label classes so live translation script can decode predictions
np.save('classes.npy', label_encoder.classes_)
print(f"Dataset size: {len(X_aug)} samples across {num_classes} classes.")

# 3. Split data (80% training, 20% validation)
X_train, X_test, y_train, y_test = train_test_split(X_aug, y_encoded, test_size=0.2, random_state=42)

# 4. Build Neural Network Architecture
print("Building Neural Network...")
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(63,)),
    
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# 5. Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train the Model
print("Starting training phase...")
history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=64,
    validation_data=(X_test, y_test),
    verbose=1
)

# 7. Evaluate on unseen test data
print("\nEvaluating model on test data...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

# 8. Save the trained model
model.save('asl_model.keras')
print("Model saved successfully as 'asl_model.keras'")