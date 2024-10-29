import cv2
import time
import av
import numpy as np
import threading
import queue

# Camera indices
cap1_index = "/dev/video0"  # First camera
cap2_index = "/dev/video2"  # Second camera

# Initialize cameras
cap1 = cv2.VideoCapture(cap1_index)
cap2 = cv2.VideoCapture(cap2_index)

if not cap1.isOpened():
    print(f"Error: Camera {cap1_index} failed to open.")
    exit()
if not cap2.isOpened():
    print(f"Error: Camera {cap2_index} failed to open.")
    exit()

# Camera settings
Width = 640  # Reduced resolution for better performance
Height = 480
FPS = 30

cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, Width)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, Height)
cap1.set(cv2.CAP_PROP_FPS, FPS)

cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, Width)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, Height)
cap2.set(cv2.CAP_PROP_FPS, FPS)

# Output dimensions for stitched frame
output_width = Width * 2  # Width of the stitched frame
output_height = Height    # Height of the stitched frame

# Display window dimensions
window_width = Width  # Width of the display window
window_height = Height

# Initial offset for panning
x_offset = 0
pan_step = 10  # Pixels to shift per arrow key press

# Variables for FPS calculation and data size
prev_time = time.time()
fps_counter = 0
total_data_size_raw = 0      # Total uncompressed data size in bytes
total_data_size_compressed = 0  # Total compressed data size in bytes

# Queues for inter-thread communication
# Set maxsize to 1 to prevent queue build-up (frame dropping)
frame1_queue = queue.Queue(maxsize=1)
frame2_queue = queue.Queue(maxsize=1)
stitched_frame_queue = queue.Queue(maxsize=1)

# Function to stitch two frames side by side
def stitch_frames(frame1, frame2):
    """Stitch two frames horizontally."""
    return cv2.hconcat([frame1, frame2])

# Initialize PyAV encoder
output_container = av.open('dummy_output.h264', mode='w', format='h264')
stream = output_container.add_stream('libx264', rate=FPS)
stream.width = output_width
stream.height = output_height
stream.pix_fmt = 'yuv420p'  # Pixel format for H.264

def capture_frames(cap, frame_queue, camera_name):
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to capture frame from {camera_name}.")
            break
        # Flip the frame upside down (vertical flip)
        frame = cv2.flip(frame, 0)
        # Try to put the frame in the queue without blocking
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # Discard the frame if the queue is full

def process_and_encode_frames():
    global total_data_size_raw, total_data_size_compressed, fps_counter
    while True:
        try:
            # Get the most recent frames
            frame1 = frame1_queue.get(timeout=1)
            frame2 = frame2_queue.get(timeout=1)
        except queue.Empty:
            continue  # Skip if any frame is not available

        # Stitch the frames together
        stitched_frame = stitch_frames(frame1, frame2)

        # Calculate data size for this frame (uncompressed)
        frame_data_size_raw = stitched_frame.size * stitched_frame.itemsize
        total_data_size_raw += frame_data_size_raw

        # Encode the frame using H.264
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(stitched_frame, cv2.COLOR_BGR2RGB)
        # Create a VideoFrame from the NumPy array
        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
        # Convert to YUV420P format
        video_frame_yuv = video_frame.reformat(format='yuv420p')
        # Encode the frame
        packets = stream.encode(video_frame_yuv)
        # Calculate compressed data size
        compressed_size = sum(packet.size for packet in packets)
        total_data_size_compressed += compressed_size

        # Put the stitched frame into the display queue
        try:
            stitched_frame_queue.put_nowait(stitched_frame)
        except queue.Full:
            pass  # Discard the frame if the queue is full

        fps_counter += 1

def display_frames():
    global x_offset
    while True:
        try:
            stitched_frame = stitched_frame_queue.get(timeout=1)
        except queue.Empty:
            continue  # Skip if no frame is available

        # Calculate maximum panning offset
        max_offset = max(0, output_width - window_width)

        # Clamp x_offset within valid range
        x_offset = max(0, min(x_offset, max_offset))

        # Display the portion of the stitched frame based on x_offset
        display_frame = stitched_frame[0:window_height, x_offset:x_offset + window_width]
        cv2.imshow('Stitched Video Feed', display_frame)

        # Handle key events for panning
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit on 'q' key
            break
        elif key == 81:  # Left arrow key
            x_offset -= pan_step
        elif key == 83:  # Right arrow key
            x_offset += pan_step

    cv2.destroyAllWindows()
    # Signal other threads to exit
    exit(0)

def calculate_fps_and_bandwidth():
    global fps_counter, total_data_size_raw, total_data_size_compressed, prev_time
    while True:
        current_time = time.time()
        elapsed_time = current_time - prev_time
        if elapsed_time >= 1.0:
            fps = fps_counter
            # Convert data sizes to megabytes (MB)
            size_raw_mb = total_data_size_raw / (1024 * 1024)
            size_compressed_mb = total_data_size_compressed / (1024 * 1024)
            # Calculate bandwidth in Mbps
            bandwidth_raw_mbps = (total_data_size_raw * 8) / (elapsed_time * 1e6)
            bandwidth_compressed_mbps = (total_data_size_compressed * 8) / (elapsed_time * 1e6)
            print(f"FPS: {fps}")
            print(f"Uncompressed Data Size: {size_raw_mb:.2f} MB, Bandwidth: {bandwidth_raw_mbps:.2f} Mbps")
            print(f"Compressed Data Size: {size_compressed_mb:.2f} MB, Bandwidth: {bandwidth_compressed_mbps:.2f} Mbps")
            fps_counter = 0
            total_data_size_raw = 0
            total_data_size_compressed = 0
            prev_time = current_time
        time.sleep(0.1)

# Start capture threads
capture_thread1 = threading.Thread(target=capture_frames, args=(cap1, frame1_queue, 'Camera 1'))
capture_thread2 = threading.Thread(target=capture_frames, args=(cap2, frame2_queue, 'Camera 2'))
capture_thread1.start()
capture_thread2.start()

# Start processing thread
processing_thread = threading.Thread(target=process_and_encode_frames)
processing_thread.start()

# Start display thread
display_thread = threading.Thread(target=display_frames)
display_thread.start()

# Start FPS calculation thread
fps_thread = threading.Thread(target=calculate_fps_and_bandwidth)
fps_thread.start()

# Wait for display thread to finish (when 'q' is pressed)
display_thread.join()

# Clean up: stop other threads
# Note: In practice, you'll need a way to signal other threads to exit gracefully.

# Flush the encoder
packets = stream.encode(None)
for packet in packets:
    total_data_size_compressed += packet.size

# Close the output container
output_container.close()

# Release camera resources
cap1.release()
cap2.release()
