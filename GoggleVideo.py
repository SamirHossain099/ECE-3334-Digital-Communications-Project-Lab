import cv2
import serial
import time

# Use the highest baud rate for maximum data transfer
ser = serial.Serial('COM4', baudrate=921600, timeout=1)

# Load the video file and reduce resolution
cap = cv2.VideoCapture("ExampleFile.mov")
target_width, target_height = 320, 240  # Higher resolution with increased baud rate

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error.")
        break

    # Resize and convert frame
    frame = cv2.resize(frame, (target_width, target_height))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, buffer = cv2.imencode('.jpg', gray_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])

    # Send frame data in larger chunks
    frame_data = buffer.tobytes()
    print("Sending frame of size:", len(frame_data))
    for i in range(0, len(frame_data), 256):  # Larger chunks
        ser.write(frame_data[i:i+256])
        time.sleep(0.01)  # Small delay to ensure stability

    time.sleep(0.03)  # Frame delay to simulate frame rate

# Release resources
cap.release()
ser.close()