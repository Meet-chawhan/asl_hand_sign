import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter
prediction_buffer = deque(maxlen=5)  # store last 5 predictions
import time




# Load the trained model
model = load_model('model.h5')



# Labels from A to Z
labels = [chr(i) for i in range(65, 91)]  # ['A', 'B', ..., 'Z']

# MediaPipe hand detector setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    base = landmarks[0]  # wrist
    landmarks -= base     # center at wrist
    max_val = np.max(np.linalg.norm(landmarks, axis=1))
    if max_val != 0:
        landmarks /= max_val  # avoid divide-by-zero
    return landmarks.flatten()

last_prediction_time = 0
prediction_interval = 0.4  # seconds
last_prediction_time = 0
prediction_interval = 0.4  # seconds
last_label = None
last_confidence = 0.0
last_prediction_time = 0
 

while True:
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks on frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract normalized landmark coordinates
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # use normalized values directly



            normalized = normalize_landmarks(landmarks)
            input_data_scaled = np.array(normalized).reshape(1, -1)



            
            # Predict
            prediction = model.predict(input_data_scaled, verbose=0)
            predicted_index = np.argmax(prediction)
            confidence = prediction[0][predicted_index]

            # Add to buffer
            prediction_buffer.append(predicted_index)

            # Majority voting
            most_common_index, count = Counter(prediction_buffer).most_common(1)[0]



            # Get predicted label safely
            if predicted_index < len(labels):
                predicted_label = labels[predicted_index]
            else:
                predicted_label = "Unknown"

            current_time = time.time()

            if confidence > 0.6:
                if current_time - last_prediction_time > prediction_interval:
                    # Ensure most_common_index is valid (e.g., between 0 and len(labels)-1)
                    if 0 <= most_common_index < len(labels):
                        last_label = labels[most_common_index]
                        last_confidence = confidence
                        last_prediction_time = current_time
                    else:
                        print("Warning: most_common_index out of range.")
            

            # Always display the last confident prediction (smooth UI)
            if last_label is not None:
                cv2.putText(frame, f'{last_label} ({last_confidence*100:.1f}%)', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)



    # Show webcam frame
    cv2.imshow("ASL A-Z Live Translator", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

