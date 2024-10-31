# threads.py

# Import Classes
from sendcontrols import Send_Control_Data
# from gyroscope import Gyroscope
from videostream import VideoStream  # Import the CameraStitcher class

# Import Threading
import threading

# Initialize components
print("Initializing components...")
send_control = Send_Control_Data()
# gyroscope = Gyroscope()
video_stream = VideoStream()  # Initialize CameraStitcher

# Start threads for each component
print("Configuring threads...")
control_thread = threading.Thread(target=send_control.start_server)
# gyroscope_thread = threading.Thread(target=gyroscope.RunGyro)
video_thread = threading.Thread(target=video_stream.run)

# Start and manage threads
print("Starting threads...")
control_thread.start()
# gyroscope_thread.start()
video_thread.start()
