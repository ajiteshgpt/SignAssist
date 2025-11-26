import os
import cv2
import time # Imported for delay

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'ok', 'not ok']
dataset_size = 50

cap = cv2.VideoCapture(0)

for j, class_name in enumerate(classes):
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(class_name))

    # --- WAITING PHASE ---
    while True:
        ret, frame = cap.read()
        
        # Add text for user instruction
        cv2.putText(frame, 'Ready? Press "Q" to Quit, "S" to Start "{}"'.format(class_name), 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        
        k = cv2.waitKey(1)
        if k == ord('s') or k == ord('S'):
            break
        # Added a safety break to quit the program entirely
        if k == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # --- CAPTURE PHASE ---
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        
        # 1. Create a copy so we don't save the text on the image
        display_frame = frame.copy() 

        # 2. Display progress on the COPY
        cv2.putText(display_frame, 'Captured: {}/{}'.format(counter + 1, dataset_size), 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('frame', display_frame)
        cv2.waitKey(1)
        
        # 3. Save the ORIGINAL clean frame
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        
        counter += 1
        
        # 4. Small sleep to allow movement variance (0.1s delay)
        time.sleep(0.05) 

cap.release()
cv2.destroyAllWindows()