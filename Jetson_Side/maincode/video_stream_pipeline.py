from capture import CameraCapture
from stitcher import stitch_frames
from encoder import FrameEncoder
from display import DisplayFeed
from utils.fps_calculator import FPSCalculator
from queues import frame1_queue, frame2_queue, stitched_frame_queue
import threading
from controls import Controls

# Camera settings
WIDTH = 640
HEIGHT = 480
FPS = 30
OUTPUT_WIDTH = WIDTH * 2
OUTPUT_HEIGHT = HEIGHT

# Initialize components
camera1 = CameraCapture('/dev/video0', frame1_queue, 'Camera 1', WIDTH, HEIGHT, FPS)
camera2 = CameraCapture('/dev/video2', frame2_queue, 'Camera 2', WIDTH, HEIGHT, FPS)
encoder = FrameEncoder(output_width=OUTPUT_WIDTH, output_height=OUTPUT_HEIGHT, fps=FPS)
display = DisplayFeed()
fps_calculator = FPSCalculator()
control = Controls()

# Start threads for each component
capture_thread1 = threading.Thread(target=camera1.capture_frames)
capture_thread2 = threading.Thread(target=camera2.capture_frames)
stitch_thread = threading.Thread(target=stitch_frames)
encode_thread = threading.Thread(target=encoder.encode_frames)
display_thread = threading.Thread(target=display.show_feed)
fps_thread = threading.Thread(target=fps_calculator.calculate_fps)
control_thread = threading.Thread(target=control.receive_joystick_data)

# Start and manage threads
capture_thread1.start()
capture_thread2.start()
stitch_thread.start()
encode_thread.start()
display_thread.start()
fps_thread.start()
control_thread.start()

# Wait for display thread to finish (when 'q' is pressed)
display_thread.join()
