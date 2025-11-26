import cv2
import mediapipe as mp

# Initialize mediapipe hand detection and drawing utils
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up webcam or video feed
cap = cv2.VideoCapture(0)

# Configure MediaPipe Hands
with mp_hands.Hands(
    max_num_hands=2,          # Maximum number of hands to detect
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the image color to RGB (required by MediaPipe)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect hand landmarks
        results = hands.process(image_rgb)
        
        # Convert the frame back to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the landmarks on the image
                mp_drawing.draw_landmarks(
                    image_bgr, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),  # Green lines
                    mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=2))  # Purple dots

        # Display the frame with hand landmarks
        cv2.imshow('Hand Detection', image_bgr)

        # Exit on pressing 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
# from flask import Flask, render_template, Response
# import cv2
# import mediapipe as mp

# app = Flask(__name__)

# # Initialize mediapipe hand detection and drawing utils
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# # OpenCV video feed
# cap = cv2.VideoCapture(0)

# def generate_frames():
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(image_rgb)
#         image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(
#                     image_bgr, 
#                     hand_landmarks, 
#                     mp_hands.HAND_CONNECTIONS,
#                     mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
#                     mp_drawing.DrawingSpec(color=(255,0,255), thickness=2, circle_radius=2))

#         ret, buffer = cv2.imencode('.jpg', image_bgr)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('1.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True)
