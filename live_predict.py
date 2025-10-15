import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter
prediction_buffer = deque(maxlen=5)  # store last 5 predictions
import time

model = load_model('model.h5')

#labeling
labels = [chr(i) for i in range(65, 91)] 

# hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

#camera
cap = cv2.VideoCapture(0)

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    base = landmarks[0]  
    landmarks -= base     
    max_val = np.max(np.linalg.norm(landmarks, axis=1))
    if max_val != 0:
        landmarks /= max_val 
    return landmarks.flatten()

last_prediction_time = 0
prediction_interval = 0.4  
last_prediction_time = 0
prediction_interval = 0.4 
last_label = None
last_confidence = 0.0
last_prediction_time = 0
 

while True:
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # left hand
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  
            normalized = normalize_landmarks(landmarks)
            input_data_scaled = np.array(normalized).reshape(1, -1)



            
            # Predictiom
            prediction = model.predict(input_data_scaled, verbose=0)
            predicted_index = np.argmax(prediction)
            confidence = prediction[0][predicted_index]

            #buffer to give stable result
            prediction_buffer.append(predicted_index)

            # majority
            most_common_index, count = Counter(prediction_buffer).most_common(1)[0]



            if predicted_index < len(labels):

                predicted_label = labels[predicted_index]
            else:
                predicted_label = "Unknown"

            current_time = time.time()

            if confidence > 0.6:
                if current_time - last_prediction_time > prediction_interval:

                    if 0 <= most_common_index < len(labels):
                        last_label = labels[most_common_index]
                        last_confidence = confidence
                        last_prediction_time = current_time
                    else:
                        print("Warning: most_common_index out of range.")

            
            if last_label is not None:
                cv2.putText(frame, f'{last_label} ({last_confidence*100:.1f}%)', (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)




    cv2.imshow("ASL A-Z Live Translator", frame)

    # q for exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

