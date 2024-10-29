import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject
import cv2
import numpy as np
import time

# Initialize GStreamer
Gst.init(None)

# Define Jetson GStreamer pipeline for dual-camera streaming with RTP
def build_pipeline():
    pipeline_str = (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw,width=640,height=480,framerate=30/1 ! "
        "videoconvert ! "
        "x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
        "rtph264pay config-interval=1 pt=96 ! "
        "udpsink host=10.161.168.162 port=5000 "
        "v4l2src device=/dev/video2 ! "
        "video/x-raw,width=640,height=480,framerate=30/1 ! "
        "videoconvert ! "
        "x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! "
        "rtph264pay config-interval=1 pt=96 ! "
        "udpsink host=10.161.168.162 port=5001 "
    )
    return Gst.parse_launch(pipeline_str)

# Run the GStreamer pipeline
pipeline = build_pipeline()
pipeline.set_state(Gst.State.PLAYING)

try:
    # Main event loop for GStreamer
    bus = pipeline.get_bus()
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
    print("Terminating...")

# Cleanup
pipeline.set_state(Gst.State.NULL)
