import queue

# Define shared queues for inter-thread communication
frame1_queue = queue.Queue(maxsize=1)
frame2_queue = queue.Queue(maxsize=1)
stitched_frame_queue = queue.Queue(maxsize=1)
