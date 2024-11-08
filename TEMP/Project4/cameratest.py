import cv2
import time
# load camera
cap = cv2.VideoCapture(0)
# stream video
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('a'):
        filename = f'saved_frame_{int(time.time())}.jpg'
        cv2.imwrite(filename, frame)
cap.release()
cv2.destroyAllWindows()