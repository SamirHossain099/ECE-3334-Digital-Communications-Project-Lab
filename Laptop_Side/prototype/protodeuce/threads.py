# # threads.py

# # Import Classes
# from sendcontrols import Send_Control_Data
# from gyroscope import Gyroscope
# from videostream import VideoStream
# from hud import HUD

# # Import Threading
# import threading

# # Initialize components
# print("Initializing components...")
# send_control = Send_Control_Data()
# gyroscope = Gyroscope()
# video_stream = VideoStream()
# hud = HUD()

# # Start threads for each component
# print("Configuring threads...")
# control_thread = threading.Thread(target=send_control.start_server)
# gyroscope_thread = threading.Thread(target=gyroscope.RunGyro)
# video_thread = threading.Thread(target=video_stream.run)
# hud_thread = threading.Thread(target=hud.run)

# # Start and manage threads
# print("Starting threads...")
# control_thread.start()
# gyroscope_thread.start()
# video_thread.start()
# hud_thread.start()

# # Optionally, wait for threads to finish (if you want to block the main thread)
# # control_thread.join()
# # gyroscope_thread.join()
# # video_thread.join()
# # threads.py

# Import Classes
from hud import HUD
from sendcontrols import Send_Control_Data
from gyroscope import Gyroscope
from videostream import VideoStream

# Import Threading and Subprocess
import threading
import subprocess
import sys
import time
import os

# Function to start GStreamer commands in separate command prompts
def start_gstreamer_commands():
    # Define the GStreamer commands
    #Samir's File Path
    # cmd1 = (
    #     'gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp, payload=96" '
    #     '! rtph264depay ! avdec_h264 ! videoconvert ! jpegenc ! '
    #     'multifilesink location="D:/Lab/Terminal1/camera1_frame_%05d.jpg"'
    # )
    
    # cmd2 = (
    #     'gst-launch-1.0 udpsrc port=5001 caps="application/x-rtp, payload=96" '
    #     '! rtph264depay ! avdec_h264 ! videoconvert ! jpegenc ! '
    #     'multifilesink location="D:/Lab/Terminal2/camera2_frame_%05d.jpg"'
    # )
    
    
    # Nick's File Path
    cmd1 = (
        'gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp, payload=96" '
        '! rtph264depay ! avdec_h264 ! videoconvert ! jpegenc ! '
        'multifilesink location="C:/temp/camera1/camera1_frame_%05d.jpg"'
    )
    
    cmd2 = (
        'gst-launch-1.0 udpsrc port=5001 caps="application/x-rtp, payload=96" '
        '! rtph264depay ! avdec_h264 ! videoconvert ! jpegenc ! '
        'multifilesink location="C:/temp/camera2/camera2_frame_%05d.jpg"'
    )
    
    # Optionally, specify the full path to gst-launch-1.0 if it's not in PATH
    # Example:
    # gst_path = "C:/gstreamer/1.0/x86_64/bin/gst-launch-1.0.exe"
    # cmd1 = f'"{gst_path}" udpsrc port=5000 caps="application/x-rtp, payload=96" ...'
    # cmd2 = f'"{gst_path}" udpsrc port=5001 caps="application/x-rtp, payload=96" ...'

    # Start Terminal 1 for Camera 1
    subprocess.Popen(
        f'cmd /c start "GStreamer_Camera1" cmd /k "{cmd1}"',
        shell=True
    )
    print("Started GStreamer command for Camera 1 in a new command prompt.")

    # Start Terminal 2 for Camera 2
    subprocess.Popen(
        f'cmd /c start "GStreamer_Camera2" cmd /k "{cmd2}"',
        shell=True
    )
    print("Started GStreamer command for Camera 2 in a new command prompt.")

    # Optionally, return the subprocess.Popen objects if you want to manage them later
    # return proc1, proc2

# Initialize components
print("Initializing components...")
send_control = Send_Control_Data()
gyroscope = Gyroscope()
video_stream = VideoStream()
hud = HUD()


# Start GStreamer commands and keep track of subprocesses
start_gstreamer_commands()

# Start threads for each component
print("Configuring threads...")
control_thread = threading.Thread(target=send_control.start_server, daemon=True)
gyroscope_thread = threading.Thread(target=gyroscope.RunGyro, daemon=True)
video_thread = threading.Thread(target=video_stream.run, daemon=True)
hud_thread = threading.Thread(target=hud.run)



# Start and manage threads
print("Starting threads...")
control_thread.start()
gyroscope_thread.start()
video_thread.start()
hud_thread.start()

# Keep the main thread alive to allow daemon threads to run
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nShutting down threads and exiting...")
    # Terminate the program gracefully
    sys.exit(0)