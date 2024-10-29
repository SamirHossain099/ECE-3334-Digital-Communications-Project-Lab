import cv2
import queue
from queues import frame1_queue, frame2_queue

class CameraCapture:
    def __init__(self, device_index, frame_queue, camera_name, width=640, height=480, fps=30):
        self.cap = cv2.VideoCapture(device_index)
        self.queue = frame_queue
        self.camera_name = camera_name
        self.width = width
        self.height = height
        self.fps = fps

        # Configure camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        if not self.cap.isOpened():
            print(f"Error: Camera {self.camera_name} (device {device_index}) failed to open.")
            exit()

    def capture_frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print(f"Error: Failed to capture frame from {self.camera_name}")
                break
            frame = cv2.flip(frame, 0)  # Flip the frame vertically
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                pass  # Discard frame if queue is full
        self.cap.release()
