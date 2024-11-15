import vlc
import time
import os

# Define video path
video_path = "Laptop_Side/prototype/ExampleFile2.mp4"

# Set VLC path
os.add_dll_directory(r'C:/Program Files/VideoLAN/VLC')

# Initialize VLC player
player = vlc.MediaPlayer(video_path)

# Start video
player.play()

# Wait until the video is finished
while player.is_playing():
    time.sleep(1)
