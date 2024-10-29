import cv2
import queue
from queues import stitched_frame_queue

class DisplayFeed:
    def __init__(self):
        self.x_offset = 0
        self.pan_step = 10
        self.window_width = 640
        self.window_height = 480
        self.output_width = 1280

    def show_feed(self):
        while True:
            try:
                stitched_frame = stitched_frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            max_offset = max(0, self.output_width - self.window_width)
            self.x_offset = max(0, min(self.x_offset, max_offset))
            display_frame = stitched_frame[0:self.window_height, self.x_offset:self.x_offset + self.window_width]
            cv2.imshow('Stitched Video Feed', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 81:
                self.x_offset -= self.pan_step
            elif key == 83:
                self.x_offset += self.pan_step
        cv2.destroyAllWindows()
