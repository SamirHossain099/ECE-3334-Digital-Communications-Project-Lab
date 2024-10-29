import cv2
import queue
from queues import frame1_queue, frame2_queue, stitched_frame_queue

def stitch_frames():
    while True:
        try:
            frame1 = frame1_queue.get(timeout=1)
            frame2 = frame2_queue.get(timeout=1)
        except queue.Empty:
            continue  # Skip if frames are not available

        # Stitch frames horizontally
        stitched_frame = cv2.hconcat([frame1, frame2])
        try:
            stitched_frame_queue.put_nowait(stitched_frame)
        except queue.Full:
            pass  # Discard frame if queue is full
