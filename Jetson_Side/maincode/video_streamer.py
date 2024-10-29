# video_streamer.py
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject
import threading

class VideoStreamer(threading.Thread):
    def __init__(self, host_ip, port1, port2):
        super().__init__()
        self.host_ip = host_ip
        self.port1 = port1
        self.port2 = port2
        self.pipeline = None
        Gst.init(None)

    def build_pipeline(self):
        pipeline_str = (
            f"v4l2src device=/dev/video0 ! "
            "video/x-raw,width=640,height=480,framerate=30/1 ! "
            "videoconvert ! "
            "x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
            f"rtph264pay config-interval=1 pt=96 ! udpsink host={self.host_ip} port={self.port1} "
            f"v4l2src device=/dev/video2 ! "
            "video/x-raw,width=640,height=480,framerate=30/1 ! "
            "videoconvert ! "
            "x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
            f"rtph264pay config-interval=1 pt=96 ! udpsink host={self.host_ip} port={self.port2} "
        )
        return Gst.parse_launch(pipeline_str)

    def run(self):
        self.pipeline = self.build_pipeline()
        self.pipeline.set_state(Gst.State.PLAYING)

        try:
            bus = self.pipeline.get_bus()
            while True:
                message = bus.timed_pop_filtered(10000, Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if message:
                    if message.type == Gst.MessageType.ERROR:
                        err, debug = message.parse_error()
                        print(f"Error: {err}, {debug}")
                        break
                    elif message.type == Gst.MessageType.EOS:
                        print("End of stream")
                        break
        except KeyboardInterrupt:
            print("Terminating Video Streamer...")

        finally:
            self.pipeline.set_state(Gst.State.NULL)
