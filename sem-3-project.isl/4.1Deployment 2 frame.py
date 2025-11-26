import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype() is deprecated")
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
import threading
import time
import logging
import pickle
from gtts import gTTS
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Adjusted for two hands detection
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7, max_num_hands=2)

labels_dict = {
    0: 'c', 1: 'a', 2: 't'
}
root = tk.Tk()
root.title("Indian Text to Speech")
root.geometry("500x400")

header_label = tk.Label(root, text="Sign Language Recognition to Speech", font=("Helvetica", 18, "bold"), fg="blue")
header_label.pack(pady=10)

text_field = tk.Text(root, height=2, width=40, font=("Helvetica", 16))
text_field.pack(pady=10)

def clear_text():
    text_field.delete('1.0', tk.END)
    logging.info('Text cleared.')

button_frame = tk.Frame(root)
button_frame.pack(pady=20)

clear_button = tk.Button(button_frame, text="Clear Text", command=clear_text, bg="orange", fg="white", font=("Helvetica", 12, "bold"), width=15)
clear_button.grid(row=0, column=0, padx=10, pady=10)

exit_button = tk.Button(button_frame, text="Exit", command=root.quit, bg="red", fg="white", font=("Helvetica", 12, "bold"), width=15)
exit_button.grid(row=0, column=1, padx=10, pady=10)

def speak_text(language='en'):
    text_to_speak = text_field.get('1.0', tk.END).strip()
    if text_to_speak:
        logging.info(f"Speaking text: {text_to_speak}")
        tts = gTTS(text=text_to_speak, lang=language)
        tts.save('output.mp3')
        os.system('start output.mp3' if os.name == 'nt' else 'mpg123 output.mp3')

speak_button = tk.Button(button_frame, text="Speak Text", command=lambda: speak_text(language='en'), bg="green", fg="white", font=("Helvetica", 12, "bold"), width=15)
speak_button.grid(row=1, column=0, padx=10, pady=10)

speak_hindi_button = tk.Button(button_frame, text="Translate Speech", command=lambda: speak_text(language='fr'), bg="purple", fg="white", font=("Helvetica", 12, "bold"), width=15)
speak_hindi_button.grid(row=1, column=1, padx=10, pady=10)

prev_prediction = None
word_count = 0
last_detected_character = None
fixed_character = ""
delayCounter = 0
start_time = time.time()

def update_text_field(text):
    if text == 'space':
        text_field.insert(tk.END, ' ')
    else:
        text_field.insert(tk.END, text)
    logging.info(f'Word added: {text if text != "space" else "space (represented as space)"}')

def run():
    global last_detected_character, fixed_character, delayCounter, start_time

    while True:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()

        if not ret:
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks for each hand
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Extract landmark data for each hand
                x_.clear()
                y_.clear()
                data_aux.clear()

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

                current_time = time.time()

                if predicted_character == last_detected_character:
                    if (current_time - start_time) >= 1.0:
                        fixed_character = predicted_character
                        if delayCounter == 0:
                            update_text_field(fixed_character)
                            delayCounter = 1
                else:
                    start_time = current_time
                    last_detected_character = predicted_character
                    delayCounter = 0

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

threading.Thread(target=run, daemon=True).start()

root.mainloop()
