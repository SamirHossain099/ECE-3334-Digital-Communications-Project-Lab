import av
import queue
from queues import stitched_frame_queue

class FrameEncoder:
    def __init__(self, output_width, output_height, fps):
        self.container = av.open('output.h264', mode='w', format='h264')
        self.stream = self.container.add_stream('libx264', rate=fps)
        self.stream.width = output_width
        self.stream.height = output_height
        self.stream.pix_fmt = 'yuv420p'

    def encode_frames(self):
        while True:
            try:
                stitched_frame = stitched_frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Convert to RGB and then YUV for encoding
            video_frame = av.VideoFrame.from_ndarray(stitched_frame, format='bgr24')
            video_frame = video_frame.reformat(format='yuv420p')
            packets = self.stream.encode(video_frame)
            for packet in packets:
                # Track compressed data size for bandwidth
                pass
        # Finalize encoding
        packets = self.stream.encode(None)
        for packet in packets:
            pass
        self.container.close()
