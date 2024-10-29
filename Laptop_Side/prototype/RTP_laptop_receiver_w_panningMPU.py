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

# Serial port settings for MPU-6050
SERIAL_PORT = 'COM3'  # Replace with your Arduino's serial port
BAUD_RATE = 115200
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# Initialize panning and display settings
camera1_folder = "D:/Lab/Terminal1/"
camera2_folder = "D:/Lab/Terminal2/"
window_width = 640
window_height = 480
x_offset = 0
max_pan_step = 20  # Adjust for panning sensitivity

# For calculating roll angle
roll = 0.0
last_time = time.time()

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

def process_mpu_data(data_line):
    global roll, last_time, x_offset

    # Parse and calculate roll angle
    try:
        values = [float(x) for x in data_line.strip().split(',')]
        if len(values) != 6:
            return

        gyro_x = values[3]  # Gyroscope X-axis data
        current_time = time.time()
        delta_time = current_time - last_time

        # Calculate roll angle
        roll += math.radians(gyro_x) * delta_time
        roll_degrees = math.degrees(roll)

        # Normalize roll angle
        if roll_degrees > 180.0:
            roll_degrees -= 360.0
        elif roll_degrees < -180.0:
            roll_degrees += 360.0

        # Control panning based on head orientation
        if roll_degrees > 15.0:
            x_offset += min(max_pan_step, roll_degrees - 15.0)
        elif roll_degrees < -15.0:
            x_offset -= min(max_pan_step, abs(roll_degrees) - 15.0)

        # Debug print statements
        print(f"Roll Degrees: {roll_degrees:.2f}, x_offset: {x_offset}")

        last_time = current_time

    except ValueError:
        pass

while True:
    # Get frames and stitch them side-by-side
    frame1 = get_latest_frame(camera1_folder, "camera1")
    frame2 = get_latest_frame(camera2_folder, "camera2")

    if frame1 is None or frame2 is None:
        print("Waiting for frames...")
        time.sleep(0.1)
        continue

    # Resize frames if necessary
    if frame1.shape != frame2.shape:
        height = min(frame1.shape[0], frame2.shape[0])
        width = min(frame1.shape[1], frame2.shape[1])
        frame1 = cv2.resize(frame1, (width, height))
        frame2 = cv2.resize(frame2, (width, height))

    stitched_frame = np.hstack((frame1, frame2))
    stitched_width = stitched_frame.shape[1]
    max_offset = max(0, stitched_width - window_width)

    # Limit x_offset within bounds
    x_offset = max(0, min(x_offset, max_offset))

    # Display panning window of the stitched frame
    display_frame = stitched_frame[0:window_height, int(x_offset):int(x_offset) + window_width]
    cv2.imshow("Head-Controlled Stitched Camera Feed", display_frame)

    # Cleanup old frames
    cleanup_old_frames(camera1_folder, "camera1")
    cleanup_old_frames(camera2_folder, "camera2")

    # Read MPU-6050 data
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        print(f"Serial Data: {line}")  # Debug: Show raw serial data
        process_mpu_data(line)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
ser.close()
