import cv2
import time

# Open video captures from both cameras
cap1 = cv2.VideoCapture(0)  # Camera 1
cap2 = cv2.VideoCapture(2)  # Camera 2

# Set the resolution of both cameras
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the stitched video width and height
output_width = 640 * 2  # Since you are stitching two 640px feeds side by side
output_height = 480

# Define the display window size (1280x720 in this case)
window_width = 640  # Horizontal portion of the video to show
window_height = 480  # Vertical size to show (since cameras are 480p)

# Initial offset for panning
x_offset = 0

# Define the amount of pixels to shift when pressing arrow keys
pan_step = 10  # Small step for smoother movement

# Variables to calculate FPS
prev_time = time.time()
fps = 0
frame_count = 0

# Main loop to capture video
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        print("Error capturing video")
        break

    # Stitch the two frames side by side
    stitched_frame = cv2.hconcat([frame1, frame2])

    # Ensure the stitched frame height matches the window height
    stitched_frame = cv2.resize(stitched_frame, (output_width, window_height))

    # Calculate the maximum offset for panning
    max_offset = max(0, output_width - window_width)

    # Ensure the offset stays within valid range
    x_offset = max(0, min(x_offset, max_offset))

    # Display only a portion of the stitched frame (panning using x_offset)
    display_frame = stitched_frame[0:window_height, x_offset:x_offset + window_width]

    # Show the frame
    cv2.imshow('Stitched Video Feed', display_frame)

    # Calculate and display FPS
    frame_count += 1
    curr_time = time.time()
    elapsed_time = curr_time - prev_time
    if elapsed_time >= 1.0:  # Update every second
        fps = frame_count / elapsed_time
        print(f"FPS: {fps:.2f}")
        frame_count = 0
        prev_time = curr_time

    # Capture key events
    key = cv2.waitKey(1) & 0xFF

    # Panning based on individual key presses
    if key == 81:  # Left arrow key
        x_offset = max(0, x_offset - pan_step)  # Pan left
        print(f"Panning left: x_offset = {x_offset}")
    elif key == 83:  # Right arrow key
        x_offset = min(max_offset, x_offset + pan_step)  # Pan right
        print(f"Panning right: x_offset = {x_offset}")
    
    # Break loop on 'q' key press
    if key == ord('q'):
        break

# Release resources
cap1.release()
cap2.release()
cv2.destroyAllWindows()
