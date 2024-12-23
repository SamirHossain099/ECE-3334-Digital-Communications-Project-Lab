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

def get_latest_frame(folder, prefix):
    """Get the latest frame in a folder with a specified prefix."""
    files = glob.glob(os.path.join(folder, f"{prefix}_frame_*.jpg"))
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    frame = cv2.imread(latest_file)
    return frame

def cleanup_old_frames(folder, prefix, max_files=10):
    """Delete older frames to manage storage."""
    files = sorted(glob.glob(os.path.join(folder, f"{prefix}_frame_*.jpg")), key=os.path.getctime)
    for f in files[:-max_files]:  # Keep only the latest `max_files` images
        os.remove(f)

camera1_folder = "D:/Lab/Terminal1/"
# camera1_folder = "C:/temp/camera1/"
camera2_folder = "D:/Lab/Terminal2/"
# camera2_folder = "C:/temp/camera2/"
while True:
    # Get the latest frames from both camera feeds
    frame1 = get_latest_frame(camera1_folder, "camera1")
    frame2 = get_latest_frame(camera2_folder, "camera2")

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
    cleanup_old_frames(camera1_folder, "camera1")
    cleanup_old_frames(camera2_folder, "camera2")

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
