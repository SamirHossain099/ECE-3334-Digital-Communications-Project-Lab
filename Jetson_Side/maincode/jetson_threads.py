# jetson_threads.py
from video_streamer import VideoStreamer
from controls import Controls
import threading

def main():
    # Set host IP and ports for video streaming
    host_ip = "10.161.168.162"
    port1 = 5000
    port2 = 5001

    # Initialize and start video streaming thread
    video_streamer = VideoStreamer(host_ip, port1, port2)
    control = Controls()

    video_streamer_thread = threading.Thread(target=video_streamer.run)
    control_thread  = threading.Thread(target=control.receive_joystick_data)

    video_streamer_thread.start()
    control_thread.start()

    # Wait for the video streaming thread to complete
    video_streamer_thread.join()

if __name__ == "__main__":
    main()
