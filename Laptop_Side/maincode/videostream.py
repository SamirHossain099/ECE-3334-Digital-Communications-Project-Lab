# videosteam.py

import cv2
import glob
import os
import time
import numpy as np

class VideoStream:
    def __init__(self, camera1_folder="C:/temp/camera1/", camera2_folder="C:/temp/camera2/"):
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

            # Display the stitched frame
            cv2.imshow("Stitched Camera Feed", stitched_frame)

            # Cleanup old frames periodically
            self.cleanup_old_frames(self.camera1_folder, "camera1")
            self.cleanup_old_frames(self.camera2_folder, "camera2")

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

        cv2.destroyAllWindows()
