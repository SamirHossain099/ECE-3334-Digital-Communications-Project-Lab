#Open two terminals and write the below commands in each terminal, make sure to have gstreamer installed
#Terminal 1
#gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp, payload=96" ! rtph264depay ! avdec_h264 ! videoconvert ! jpegenc ! multifilesink location="D:/Lab/Terminal1/camera1_frame_%05d.jpg"

#Terminal 2
#gst-launch-1.0 udpsrc port=5001 caps="application/x-rtp, payload=96" ! rtph264depay ! avdec_h264 ! videoconvert ! jpegenc ! multifilesink location="D:/Lab/Terminal2/camera2_frame_%05d.jpg"

### Make sure to change file save and retrive location so at the end of two ^ Terminal commands change save directory
### Below camera1_folder, camera2_folder edit the retrive directories

import serial
import time
import threading
import cv2
import glob
import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk

# Serial port settings (adjust the port name as needed)
SERIAL_PORT = 'COM3'  # Replace with your Arduino's serial port
BAUD_RATE = 115200

# Initialize serial connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    ser.reset_input_buffer()  # Clear buffer after opening
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit(1)

# Global variables
roll_angle = 0.0  # For MPU-6050 data
stitched_frame = None  # For the stitched video frame

# Locks for thread-safe access
roll_lock = threading.Lock()
frame_lock = threading.Lock()

def reset_mpu6050():
    """Send a killswitch command to reset MPU-6050."""
    ser.write(b'`')
    print("Killswitch activated. Resetting MPU-6050...")
    time.sleep(1)
    ser.reset_input_buffer()  # Clear the serial buffer to start fresh

def process_data(data_line):
    """Parse serial data and extract roll angle."""
    global roll_angle

    try:
        # Extract the roll angle from the data_line
        if 'Roll Angle:' in data_line:
            # Split the line to extract the angle
            parts = data_line.split('Roll Angle:')
            if len(parts) == 2:
                angle_part = parts[1].split('degrees')[0].strip()
                roll_degrees = float(angle_part)
                with roll_lock:
                    roll_angle = roll_degrees
                print(f"Processed Roll Angle: {roll_degrees:.2f} degrees")  # Debug
            else:
                print(f"Unexpected data format: {data_line}")
        else:
            print(f"Unexpected data format: {data_line}")

    except ValueError as e:
        print(f"Data conversion error: {e}, Data received: {data_line}")

def data_collection_thread():
    """Thread function for collecting MPU-6050 data."""
    print("Starting data collection in 5 seconds...")
    time.sleep(5)
    print("Data collection started.")

    try:
        while True:
            # Read data from MPU-6050
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"Raw data: {line}")  # Debug: show raw serial data
                if line:
                    process_data(line)
            else:
                time.sleep(0.01)  # Small delay to prevent 100% CPU usage

    except Exception as e:
        print(f"Data collection error: {e}")
    finally:
        if ser.is_open:
            ser.close()

def video_stream_thread():
    """Thread function to handle video streaming and stitching."""
    global stitched_frame

    # Video stream code
    import time

    def get_latest_frame(folder, prefix):
        """Get the latest frame in a folder with a specified prefix."""
        files = glob.glob(os.path.join(folder, f"{prefix}_frame_*.jpg"))
        if not files:
            return None
        latest_file = max(files, key=os.path.getctime)
        frame = cv2.imread(latest_file)
        return frame

    def cleanup_old_frames(folder, prefix, max_files=10):
        """Delete older frames to manage storage."""
        files = sorted(glob.glob(os.path.join(folder, f"{prefix}_frame_*.jpg")), key=os.path.getctime)
        for f in files[:-max_files]:  # Keep only the latest `max_files` images
            os.remove(f)

    camera1_folder = "D:/Lab/Terminal1/"
    # camera1_folder = "C:/temp/camera1/"
    camera2_folder = "D:/Lab/Terminal2/"
    # camera2_folder = "C:/temp/camera2/"

    while True:
        # Get the latest frames from both camera feeds
        frame1 = get_latest_frame(camera1_folder, "camera1")
        frame2 = get_latest_frame(camera2_folder, "camera2")

        if frame1 is None or frame2 is None:
            print("Waiting for frames...")
            time.sleep(0.1)
            continue

        # Resize frames if necessary to ensure they match
        if frame1.shape != frame2.shape:
            height = min(frame1.shape[0], frame2.shape[0])
            width = min(frame1.shape[1], frame2.shape[1])
            frame1 = cv2.resize(frame1, (width, height))
            frame2 = cv2.resize(frame2, (width, height))

        # Stitch frames side by side
        stitched = np.hstack((frame1, frame2))

        # Resize stitched frame to a consistent size if needed
        # For example, resize to (1280, 480)
        stitched = cv2.resize(stitched, (1280, 480))

        # Update the global stitched_frame variable
        with frame_lock:
            stitched_frame = stitched.copy()

        # Cleanup old frames periodically
        cleanup_old_frames(camera1_folder, "camera1")
        cleanup_old_frames(camera2_folder, "camera2")

        # Small delay to prevent excessive CPU usage
        time.sleep(0.01)

def main():
    """Main function to start data collection, video stream, and GUI."""
    global stitched_frame

    # Start the data collection thread
    data_thread = threading.Thread(target=data_collection_thread, daemon=True)
    data_thread.start()

    # Start the video streaming thread
    video_thread = threading.Thread(target=video_stream_thread, daemon=True)
    video_thread.start()

    # Set up the GUI in the main thread
    root = tk.Tk()
    root.title("Panoramic Video Viewer")

    # Create a canvas to display the video
    canvas_width = 640
    canvas_height = 480
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
    canvas.pack()

    # Placeholder image to initialize the canvas
    placeholder_image = Image.new("RGB", (canvas_width, canvas_height))
    photo = ImageTk.PhotoImage(placeholder_image)
    image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo  # Keep a reference to prevent garbage collection

    def update_image():
        with frame_lock:
            current_frame = stitched_frame.copy() if stitched_frame is not None else None

        if current_frame is not None:
            # Convert OpenCV image (BGR) to PIL Image (RGB)
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(current_frame)

            # Get the roll angle safely
            with roll_lock:
                angle = roll_angle  # Get the current roll angle

            # Clamp the roll angle to -90 to +90 degrees
            angle = max(min(angle, 90), -90)

            # Map the roll angle to x_offset
            frame_width = pil_image.width
            view_width = canvas_width
            max_offset = frame_width - view_width

            # Map roll angle to x_offset
            x_offset = int(((-angle + 90) / 180) * max_offset)

            # Ensure x_offset is within bounds
            x_offset = max(0, min(x_offset, max_offset))

            # Debug: Print roll_angle and x_offset
            print(f"Update Image - Roll Angle: {angle:.2f}, X Offset: {x_offset}")  # Debug

            # Crop the image to the current view
            box = (x_offset, 0, x_offset + view_width, pil_image.height)
            cropped_image = pil_image.crop(box)

            # Convert the cropped image to PhotoImage
            photo = ImageTk.PhotoImage(cropped_image)

            # Update the image on the canvas
            canvas.itemconfig(image_on_canvas, image=photo)
            canvas.image = photo  # Update the reference
        else:
            # No frame available, keep placeholder or display a message
            pass

        # Schedule the next update
        root.after(50, update_image)  # Update every 50 milliseconds

    # Start the image update loop
    update_image()

    # Start the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    main()
