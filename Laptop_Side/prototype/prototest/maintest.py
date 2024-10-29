import threading
import time
from gyroscope import Gyroscope
from video_display import VideoDisplay

# Shared orientation variable and a lock for synchronization
orientation_lock = threading.Lock()
orientation = "Looking Forward"

# Functions to update and read orientation
def update_orientation(new_orientation):
    global orientation
    with orientation_lock:
        orientation = new_orientation

def get_orientation():
    with orientation_lock:
        return orientation

# Modify the Gyroscope class to call update_orientation
class GyroscopeModified(Gyroscope):
    def process_data(self, data_line):
        super().process_data(data_line)  # Original processing
        update_orientation(self.orientation)  # Update global orientation

# Initialize components
print("Initializing components...")
gyroscope = GyroscopeModified()
video_display = VideoDisplay(get_orientation)

# Start threads for each component
print("Configuring threads...")
gyroscope_thread = threading.Thread(target=gyroscope.RunGyro)
video_thread = threading.Thread(target=video_display.display_stitched_feed)

# Start threads
print("Starting threads...")
gyroscope_thread.start()
video_thread.start()

try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Terminating...")
finally:
    # Cleanup if needed
    pass
