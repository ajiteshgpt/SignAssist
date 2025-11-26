import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Verify directories exist
if not os.path.exists(DATA_DIR):
    print("Error: Data directory not found.")
    exit()

for dir_ in os.listdir(DATA_DIR):
    # Skip hidden files like .DS_Store
    if dir_.startswith('.'):
        continue
        
    print(f"Processing class: {dir_}")
    
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        
        # Safety check: Ensure image loaded correctly
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            # We only intend to train on ONE hand per image for now.
            # Taking [0] prevents errors if a second hand accidentally appears in background.
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # 1. Collect all X and Y coordinates
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            # 2. Normalize the coordinates (Shift to 0,0)
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                
                # Subtracting the minimum value centers the hand
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

# Save the data
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()

print(f"Done! Processed {len(data)} samples.")