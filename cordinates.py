import cv2
import mediapipe as mp
import os
import csv

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Set your image folder path
IMAGE_FOLDER ='asl_alphabet_train'   # Change this to your image folder
CSV_OUTPUT = 'hand_landmarks.csv'

# Prepare CSV
header = ['label']
for i in range(21):  # 21 landmarks
    header += [f'x{i}', f'y{i}', f'z{i}']

with open(CSV_OUTPUT, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for label_folder in os.listdir(IMAGE_FOLDER):
        label_path = os.path.join(IMAGE_FOLDER, label_folder)

        if not os.path.isdir(label_path):
            continue

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    row = [label_folder]
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                    writer.writerow(row)

hands.close()
print(f"Landmark coordinates saved to {CSV_OUTPUT}")
