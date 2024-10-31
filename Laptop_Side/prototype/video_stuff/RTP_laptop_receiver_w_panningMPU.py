#Open two terminals and write the below commands in each terminal, make sure to have gstreamer installed
#Terminal 1
#gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp, payload=96" ! rtph264depay ! avdec_h264 ! videoconvert ! jpegenc ! multifilesink location="D:/Lab/Terminal1/camera1_frame_%05d.jpg"

#Terminal 2
#gst-launch-1.0 udpsrc port=5001 caps="application/x-rtp, payload=96" ! rtph264depay ! avdec_h264 ! videoconvert ! jpegenc ! multifilesink location="D:/Lab/Terminal2/camera2_frame_%05d.jpg"

### Make sure to change file save and retrive location so at the end of two ^ Terminal commands change save directory
### Below camera1_folder, camera2_folder edit the retrive directories

import cv2
import glob
import os
import time
import numpy as np
import serial
import math

# Serial port settings
SERIAL_PORT = 'COM3'  # Update with your port
BAUD_RATE = 115200

# Initialize serial connection for the MPU
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# Variables for panning and roll calculation
roll = 0.0
last_time = time.time()
window_width = 640
window_height = 480
x_offset = 0
stitched_frame = None

# MPU threshold for panning
pan_sensitivity = 1.5  # Higher values reduce panning sensitivity
max_offset = 0  # This will be set once frames are loaded

camera1_folder = "D:/Lab/Terminal1/"
camera2_folder = "D:/Lab/Terminal2/"

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

def process_data(data_line):
    """Parse serial data and calculate roll."""
    global roll, last_time, x_offset, max_offset

    try:
        values = [float(x) for x in data_line.strip().split(',')]
        if len(values) != 6:
            return

        accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = values
        current_time = time.time()
        delta_time = current_time - last_time

        # Calculate roll angle from gyro data
        roll += math.radians(gyro_x) * delta_time
        roll_degrees = math.degrees(roll)

        # Normalize roll angle to -180 to 180 degrees
        roll_degrees = (roll_degrees + 180) % 360 - 180

        # Adjust x_offset based on roll_degrees
        if roll_degrees > pan_sensitivity:
            x_offset += int((roll_degrees - pan_sensitivity) * 2)  # Adjust scale as needed
        elif roll_degrees < -pan_sensitivity:
            x_offset -= int((-roll_degrees - pan_sensitivity) * 2)

        # Ensure the offset stays within valid range
        x_offset = max(0, min(x_offset, max_offset))

        last_time = current_time

    except ValueError:
        print("Data conversion error.")

# Main loop for video display
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
    stitched_width = stitched_frame.shape[1]
    max_offset = max(0, stitched_width - window_width)

    # Extract the panning window from the stitched frame
    display_frame = stitched_frame[0:window_height, x_offset:x_offset + window_width]

    # Display the panning window
    cv2.imshow("Panning Stitched Camera Feed", display_frame)

    # Cleanup old frames periodically
    cleanup_old_frames(camera1_folder, "camera1")
    cleanup_old_frames(camera2_folder, "camera2")

    # Read and process gyro data for panning
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        process_data(line)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
ser.close()
