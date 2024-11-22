#Open two terminals and write the below commands in each terminal, make sure to have gstreamer installed
#Terminal 1
#gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp, payload=96" ! rtph264depay ! avdec_h264 ! videoconvert ! jpegenc ! multifilesink location="D:/Lab/Terminal1/camera1_frame_%05d.jpg"
#gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp, payload=96" ! rtph264depay ! avdec_h264 ! videoconvert ! jpegenc ! multifilesink location="C:/temp/camera1/camera1_frame_%05d.jpg"

#Terminal 2
#gst-launch-1.0 udpsrc port=5001 caps="application/x-rtp, payload=96" ! rtph264depay ! avdec_h264 ! videoconvert ! jpegenc ! multifilesink location="D:/Lab/Terminal2/camera2_frame_%05d.jpg"
#gst-launch-1.0 udpsrc port=5001 caps="application/x-rtp, payload=96" ! rtph264depay ! avdec_h264 ! videoconvert ! jpegenc ! multifilesink location="C:/temp/camera2/camera2_frame_%05d.jpg"


### Make sure to change file save and retrive location so at the end of two ^ Terminal commands change save directory
### Below camera1_folder, camera2_folder edit the retrive directories

import cv2
import glob
import os
import time
import numpy as np
import shared_data  # Import shared_data to access roll_angle
from hud import HUD  # Import the HUD class

class VideoStream:
    # def __init__(self, camera1_folder="C:/temp/camera1/", camera2_folder="C:/temp/camera2/"):
    def __init__(self, camera1_folder="D:/Lab/Terminal1/", camera2_folder="D:/Lab/Terminal2/"):

        self.camera1_folder = camera1_folder
        self.camera2_folder = camera2_folder
        self.running = True
        self.hud = HUD()  # Initialize the HUD

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

            # Process frames (e.g., stitching, overlaying HUD)
            stitched_frame = self.stitch_frames(frame1, frame2)
            angle = shared_data.roll_angle  # Get the roll angle from shared data
            angle = max(min(angle, 90), -90)
            frame_width = stitched_frame.shape[1]
            view_width = 640
            max_offset = frame_width - view_width
            x_offset = int(((-angle + 90) / 180) * max_offset)
            x_offset = max(0, min(x_offset, max_offset))
            y_start = 0
            y_end = stitched_frame.shape[0]
            x_start = x_offset
            x_end = x_offset + view_width
            cropped_frame = stitched_frame[y_start:y_end, x_start:x_end]
            cropped_frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            hud_frame = self.hud.draw_on_frame(cropped_frame_rgb)
            hud_frame_bgr = cv2.cvtColor(hud_frame, cv2.COLOR_RGB2BGR)

            # Display the frame with HUD overlay
            cv2.imshow("Panned Camera Feed with HUD", hud_frame_bgr)

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
