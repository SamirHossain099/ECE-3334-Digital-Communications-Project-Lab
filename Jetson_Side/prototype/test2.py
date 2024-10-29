import cv2
import numpy as np
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open video stream")
else:
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        cv2.imshow('video', frame)

        frame_size = frame.nbytes
        resolution = np.shape(frame)
        print("Frame Size: ", frame_size)
        print("Frame Resolution: ", resolution)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

