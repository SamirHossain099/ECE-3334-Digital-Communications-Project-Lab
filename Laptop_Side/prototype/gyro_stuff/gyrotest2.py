import cv2
import time
import math
import serial
import numpy as np

# Serial port settings (adjust as needed)
SERIAL_PORT = 'COM3'
BAUD_RATE = 115200
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# Image path
IMAGE_PATH = r'C:\Users\samir\Downloads\1280x480.jpg'

# Load the panoramic image
image = cv2.imread(IMAGE_PATH)
if image is None:
    print("Failed to load image.")
    exit()

# Check image dimensions
height, width, _ = image.shape
if width != 1280 or height != 480:
    print("Image size must be 1280x480.")
    exit()

# Display dimensions
display_width = 640
display_height = 480

# Initial x_offset to start from the middle
x_offset = (width - display_width) // 2

# Set sensitivity for panning adjustments
panning_step = 15  # Adjusts how much we pan on each update
roll_threshold = 10  # Minimum roll degrees to start panning

def process_data(data_line):
    """Parse serial data and return roll angle."""
    try:
        values = [float(x) for x in data_line.strip().split(',')]
        if len(values) != 6:
            return None

        # Extract gyroscope X-axis for roll calculation
        _, _, _, gyro_x, _, _ = values

        # Convert gyro_x to roll in degrees (simulate small movements for test)
        roll = math.degrees(math.radians(gyro_x))
        return roll

    except ValueError:
        return None

try:
    while True:
        # Read a line from MPU-6050
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()
            roll = process_data(line)

            if roll is not None:
                print(f"Detected roll: {roll:.2f} degrees")  # Debug info

                # Adjust x_offset based on roll
                if roll > roll_threshold:  # Panning Right
                    x_offset += panning_step
                    print("Panning Right")
                elif roll < -roll_threshold:  # Panning Left
                    x_offset -= panning_step
                    print("Panning Left")

                # Enforce x_offset bounds
                x_offset = max(0, min(x_offset, width - display_width))
                print(f"x_offset: {x_offset}")  # Debug info

            # Crop and display the current view of the panorama
            display_image = image[0:display_height, x_offset:x_offset + display_width]
            cv2.imshow("Panoramic View", display_image)

            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:
    ser.close()
    cv2.destroyAllWindows()
