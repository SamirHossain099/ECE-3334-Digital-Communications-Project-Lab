# videostream.py

import cv2
import glob
import os
import time
import numpy as np
import shared_data  # Import shared_data to access roll_angle

class VideoStream:
    def __init__(self, camera1_folder="C:/temp/camera1/", camera2_folder="C:/temp/camera2/"):
    # def __init__(self, camera1_folder="D:/Lab/Terminal1/", camera2_folder="D:/Lab/Terminal2/"):
        self.camera1_folder = camera1_folder
        self.camera2_folder = camera2_folder
        self.running = True

    def get_latest_frame(self, folder, prefix):
        """Get the latest frame in a folder with a specified prefix."""
        files = glob.glob(os.path.join(folder, f"{prefix}_frame_*.jpg"))
        if not files:
            return None
        latest_file = max(files, key=os.path.getctime)
        frame = cv2.imread(latest_file)
        return frame

    def cleanup_old_frames(self, folder, prefix, max_files=10):
        """Delete older frames to manage storage."""
        files = sorted(glob.glob(os.path.join(folder, f"{prefix}_frame_*.jpg")), key=os.path.getctime)
        for f in files[:-max_files]:  # Keep only the latest `max_files` images
            os.remove(f)

    def run(self):
        while self.running:
            # Get the latest frames from both camera feeds
            frame1 = self.get_latest_frame(self.camera1_folder, "camera1")
            frame2 = self.get_latest_frame(self.camera2_folder, "camera2")

            if frame1 is None or frame2 is None:
                print("Waiting for frames...")
                time.sleep(0.1)
                continue

            # Resize frames if necessary to ensure they match
            if frame1.shape != frame2.shape:
                height = min(frame1.shape[0], frame2.shape[0])
                width = min(frame1.shape[1], frame2.shape[1])
                frame1 = cv2.resize(frame1, (width, height))
                frame2 = cv2.resize(frame2, (width, height))

            # Stitch frames side by side
            stitched_frame = np.hstack((frame1, frame2))

            # Resize stitched frame to consistent size, e.g., (1280, 480)
            stitched_frame = cv2.resize(stitched_frame, (1280, 480))

            # Get roll angle safely
            with shared_data.roll_lock:
                angle = shared_data.roll_angle

            # Clamp the roll angle to -90 to +90 degrees
            angle = max(min(angle, 90), -90)

            # Map the roll angle to x_offset
            frame_width = stitched_frame.shape[1]  # Width of stitched frame
            view_width = 640  # Width of the display window
            max_offset = frame_width - view_width

            # Map roll angle to x_offset
            x_offset = int(((-angle + 90) / 180) * max_offset)

            # Ensure x_offset is within bounds
            x_offset = max(0, min(x_offset, max_offset))

            # Debug: Print roll_angle and x_offset
            print(f"Update Image - Roll Angle: {angle:.2f}, X Offset: {x_offset}")  # Debug

            # Crop the image to the current view
            y_start = 0  # Assuming full height
            y_end = stitched_frame.shape[0]
            x_start = x_offset
            x_end = x_offset + view_width
            cropped_frame = stitched_frame[y_start:y_end, x_start:x_end]

            # Display the cropped frame
            cv2.imshow("Panned Camera Feed", cropped_frame)

            # Cleanup old frames periodically
            self.cleanup_old_frames(self.camera1_folder, "camera1")
            self.cleanup_old_frames(self.camera2_folder, "camera2")

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

            # Small delay to prevent high CPU usage
            time.sleep(0.01)

        cv2.destroyAllWindows()
