# maintest.py
import threading
import time
from video_display import VideoDisplay
from gyro_reader import GyroReader

# Initialize the gyro reader
gyro_reader = GyroReader(serial_port='COM3', baud_rate=115200)

# Define a callback function for the VideoDisplay to get orientation
def get_orientation_callback():
    return gyro_reader.get_orientation()

# Initialize the video display with the gyro callback
video_display = VideoDisplay(get_orientation_callback)

# Create threads for each component
gyro_thread = threading.Thread(target=gyro_reader.run)
video_thread = threading.Thread(target=video_display.display_stitched_feed)

# Start both threads
gyro_thread.start()
video_thread.start()

# Wait for both threads to complete
gyro_thread.join()
video_thread.join()
