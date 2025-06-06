import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    base = landmarks[0]  # wrist
    landmarks -= base     # center at wrist
    max_val = np.max(np.linalg.norm(landmarks, axis=1))
    landmarks /= max_val  # scale normalization
    return landmarks.flatten()

# -----------------------------
# AUGMENTATION FUNCTIONS
# -----------------------------
def add_noise(X, noise_level=0.01):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def rotate_2d_points(X, angle_degrees):
    angle = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    X_rot = X.copy()
    for i in range(0, X.shape[1], 3):
        x = X[:, i]
        y = X[:, i+1]
        X_rot[:, i] = cos_a * x - sin_a * y
        X_rot[:, i+1] = sin_a * x + cos_a * y
        # Z stays the same
    return X_rot

def scale_landmarks(X, scale=1.05):
    X_scaled = X.copy()
    for i in range(0, X.shape[1], 3):
        X_scaled[:, i] *= scale
        X_scaled[:, i+1] *= scale
    return X_scaled

# -----------------------------
# LOAD & PREPARE DATA
# -----------------------------
df = pd.read_csv('hand_landmarks.csv')
X_raw = df.drop(columns=['label']).values
X = np.array([normalize_landmarks(row) for row in X_raw])

y_str = df['label'].values

X = np.nan_to_num(X)

label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y_str)
num_classes = len(label_encoder.classes_)
y_cat = to_categorical(y_int, num_classes=num_classes)

# -----------------------------
# AUGMENTATION
# -----------------------------
X_aug1 = add_noise(X, 0.01)
X_aug2 = rotate_2d_points(X, 5)
X_aug3 = scale_landmarks(X, 1.03)

X_combined = np.vstack([X, X_aug1, X_aug2, X_aug3])
y_combined = np.vstack([y_cat, y_cat, y_cat, y_cat])

# -----------------------------
# SPLIT & TRAIN MODEL
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42)

model = Sequential([
    Dense(256, activation='relu', input_shape=(63,)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train
history = model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=32,
                    validation_split=0.1)

# Save model and encoder
model.save('model.h5')

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy*100:.2f}%")

