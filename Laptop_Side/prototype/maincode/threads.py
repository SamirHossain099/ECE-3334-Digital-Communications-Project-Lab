# threads.py

# Import Classes
from sendcontrols import Send_Control_Data
from gyroscope import Gyroscope
from videostream import VideoStream
from hud import HUD

# Import Threading
import threading

# Initialize components
print("Initializing components...")
send_control = Send_Control_Data()
gyroscope = Gyroscope()
video_stream = VideoStream()
hud = HUD()

# Start threads for each component
print("Configuring threads...")
control_thread = threading.Thread(target=send_control.start_server)
gyroscope_thread = threading.Thread(target=gyroscope.RunGyro)
video_thread = threading.Thread(target=video_stream.run)
hud_thread = threading.Thread(target=hud.run)

# Start and manage threads
print("Starting threads...")
control_thread.start()
gyroscope_thread.start()
video_thread.start()
hud_thread.start()

# Optionally, wait for threads to finish (if you want to block the main thread)
# control_thread.join()
# gyroscope_thread.join()
# video_thread.join()
