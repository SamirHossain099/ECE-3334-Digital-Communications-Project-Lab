#Open two terminals and write the below commands in each terminal, make sure to have gstreamer installed
#Terminal 1
#gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp, payload=96" ! rtph264depay ! avdec_h264 ! videoconvert ! jpegenc ! multifilesink location="D:/Lab/Terminal1/camera1_frame_%05d.jpg"

#Terminal 2
#gst-launch-1.0 udpsrc port=5001 caps="application/x-rtp, payload=96" ! rtph264depay ! avdec_h264 ! videoconvert ! jpegenc ! multifilesink location="D:/Lab/Terminal2/camera2_frame_%05d.jpg"

### Make sure to change file save and retrive location so at the end of two ^ Terminal commands change save directory
### Below camera1_folder, camera2_folder edit the retrive directories

# video_display.py
import cv2
import numpy as np
import glob
import os
import time

class VideoDisplay:
    def __init__(self, get_orientation_callback, width=640, height=480, pan_step=50):
        self.get_orientation = get_orientation_callback
        self.stitched_width = width * 2
        self.height = height
        self.display_width = width
        self.pan_step = pan_step
        self.x_offset = (self.stitched_width - self.display_width) // 2  # Center

    def get_latest_frame(self, folder, prefix):
        files = glob.glob(os.path.join(folder, f"{prefix}_frame_*.jpg"))
        if not files:
            return None
        latest_file = max(files, key=os.path.getctime)
        frame = cv2.imread(latest_file)
        return frame

    def display_stitched_feed(self):
        camera1_folder = "D:/Lab/Terminal1/"
        camera2_folder = "D:/Lab/Terminal2/"

        while True:
            frame1 = self.get_latest_frame(camera1_folder, "camera1")
            frame2 = self.get_latest_frame(camera2_folder, "camera2")

            if frame1 is None or frame2 is None:
                print("Waiting for frames...")
                time.sleep(0.1)
                continue

            frame1 = cv2.resize(frame1, (self.display_width, self.height))
            frame2 = cv2.resize(frame2, (self.display_width, self.height))
            stitched_frame = np.hstack((frame1, frame2))

            # Get orientation and adjust the view accordingly
            orientation = self.get_orientation()
            if orientation == "Looking Left":
                self.pan_left()
            elif orientation == "Looking Right":
                self.pan_right()
            else:
                self.center_view()

            display_frame = stitched_frame[0:self.height, self.x_offset:self.x_offset + self.display_width]
            cv2.imshow("Stitched Camera Feed", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def pan_left(self):
        self.x_offset = max(0, self.x_offset - self.pan_step)

    def pan_right(self):
        max_offset = self.stitched_width - self.display_width
        self.x_offset = min(max_offset, self.x_offset + self.pan_step)

    def center_view(self):
        self.x_offset = (self.stitched_width - self.display_width) // 2
