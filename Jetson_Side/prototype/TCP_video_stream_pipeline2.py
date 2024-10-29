import cv2
import time
import av
import numpy as np
import threading
import queue
import socket
import struct

# Camera indices
cap1_index = "/dev/video0"  # First camera
cap2_index = "/dev/video2"  # Second camera

# TCP Configuration for streaming to laptop
TCP_IP = "10.161.165.25"  # Replace with the laptop's IP
TCP_PORT = 5005

# Initialize TCP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((TCP_IP, TCP_PORT))

# Camera settings
Width, Height, FPS = 640, 480, 30

# Initialize cameras
cap1 = cv2.VideoCapture(cap1_index)
cap2 = cv2.VideoCapture(cap2_index)

# Set camera properties for both cameras
for cap in (cap1, cap2):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Height)
    cap.set(cv2.CAP_PROP_FPS, FPS)

# Output dimensions for stitched frame
output_width, output_height = Width * 2, Height

# Queues for inter-thread communication
frame1_queue, frame2_queue = queue.Queue(maxsize=1), queue.Queue(maxsize=1)

# Initialize PyAV encoder
output_container = av.open('dummy_output.h264', mode='w', format='h264')
stream = output_container.add_stream('libx264', rate=FPS)
stream.width, stream.height, stream.pix_fmt = output_width, output_height, 'yuv420p'

# Capture frames from both cameras
def capture_frames(cap, frame_queue, camera_name):
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Failed to capture frame from {camera_name}.")
            break
        # Flip the frame if needed and enqueue
        frame = cv2.flip(frame, 0)
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass  # Discard if queue is full

# Process and send frames over TCP
def process_and_encode_frames():
    while True:
        try:
            # Get frames from each camera queue
            frame1 = frame1_queue.get(timeout=1)
            frame2 = frame2_queue.get(timeout=1)
        except queue.Empty:
            continue

        # Stitch frames and encode
        stitched_frame = cv2.hconcat([frame1, frame2])
        frame_rgb = cv2.cvtColor(stitched_frame, cv2.COLOR_BGR2RGB)
        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
        video_frame_yuv = video_frame.reformat(format='yuv420p')
        packets = stream.encode(video_frame_yuv)

        for packet in packets:
            packet_bytes = bytes(packet)
            packet_size = len(packet_bytes)
            try:
                # Send packet size followed by packet data
                sock.sendall(struct.pack("I", packet_size) + packet_bytes)
            except Exception as e:
                print(f"Failed to send packet: {e}")
                break

# Start threads
capture_thread1 = threading.Thread(target=capture_frames, args=(cap1, frame1_queue, 'Camera 1'))
capture_thread2 = threading.Thread(target=capture_frames, args=(cap2, frame2_queue, 'Camera 2'))
processing_thread = threading.Thread(target=process_and_encode_frames)

capture_thread1.start()
capture_thread2.start()
processing_thread.start()

# Wait for threads to finish
capture_thread1.join()
capture_thread2.join()
processing_thread.join()

# Close resources
cap1.release()
cap2.release()
output_container.close()
sock.close()
