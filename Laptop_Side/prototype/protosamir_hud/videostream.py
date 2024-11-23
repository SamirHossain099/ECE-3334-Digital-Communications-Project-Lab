# videostream.py

import cv2
import glob
import os
import time
import numpy as np
import shared_data  # Import shared_data to access roll_angle, steering, and throttle

class VideoStream:
    # def __init__(self, camera1_folder="C:/temp/camera1/", camera2_folder="C:/temp/camera2/"):
    def __init__(self, camera1_folder="D:/Lab/Terminal1/", camera2_folder="D:/Lab/Terminal2/"):
        self.camera1_folder = camera1_folder
        self.camera2_folder = camera2_folder
        self.running = True

    def get_latest_frame(self, folder, prefix):
        """Get the latest frame in a folder with a specified prefix."""
        files = glob.glob(os.path.join(folder, f"{prefix}_frame_*.jpg"))
        if not files:
            return None
        latest_file = max(files, key=os.path.getctime)
        frame = cv2.imread(latest_file)
        return frame

    def cleanup_old_frames(self, folder, prefix, max_files=10):
        """Delete older frames to manage storage."""
        files = sorted(glob.glob(os.path.join(folder, f"{prefix}_frame_*.jpg")), key=os.path.getctime)
        for f in files[:-max_files]:  # Keep only the latest `max_files` images
            os.remove(f)

    def run(self):
        while self.running:
            # Get the latest frames from both camera feeds
            frame1 = self.get_latest_frame(self.camera1_folder, "camera1")
            frame2 = self.get_latest_frame(self.camera2_folder, "camera2")

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
            stitched_frame = np.hstack((frame1, frame2))

            # Resize stitched frame to consistent size, e.g., (1280, 480)
            stitched_frame = cv2.resize(stitched_frame, (1280, 480))

            # Get roll angle safely
            with shared_data.roll_lock:
                angle = shared_data.roll_angle

            # Clamp the roll angle to -90 to +90 degrees
            angle = max(min(angle, 90), -90)

            # Map the roll angle to x_offset
            frame_width = stitched_frame.shape[1]  # Width of stitched frame
            view_width = 640  # Width of the display window
            max_offset = frame_width - view_width

            # Map roll angle to x_offset
            x_offset = int(((-angle + 90) / 180) * max_offset)

            # Ensure x_offset is within bounds
            x_offset = max(0, min(x_offset, max_offset))

            # Debug: Print roll_angle and x_offset
            print(f"Update Image - Roll Angle: {angle:.2f}, X Offset: {x_offset}")  # Debug

            # Crop the image to the current view
            y_start = 0  # Assuming full height
            y_end = stitched_frame.shape[0]
            x_start = x_offset
            x_end = x_offset + view_width
            cropped_frame = stitched_frame[y_start:y_end, x_start:x_end]

            cropped_frame_resize = cv2.resize(cropped_frame, (1280, 720))  # Adjust as needed

            # Overlay HUD on the cropped_frame_resize
            hud_frame = self.add_hud(cropped_frame_resize)

            # Display the frame with HUD
            cv2.imshow("Panned Camera Feed with HUD", hud_frame)

            # Cleanup old frames periodically
            self.cleanup_old_frames(self.camera1_folder, "camera1")
            self.cleanup_old_frames(self.camera2_folder, "camera2")

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

            # Small delay to prevent high CPU usage
            time.sleep(0.01)

        cv2.destroyAllWindows()

    def add_hud(self, frame):
        """Add HUD elements to the frame."""
        # Retrieve steering and throttle values safely
        with shared_data.steering_lock:
            steering = shared_data.steering_value
        with shared_data.throttle_lock:
            throttle = shared_data.throttle_value

        # Map throttle to speed (0-60)
        # throttle_value ranges from 2 (no speed) to 0 (max speed)
        speed = ((2.0 - throttle) / 2.0) * 60.0  # Linear mapping

        # Map steering to wheel position
        # steering_value ranges from 0.0 (full left) to 2.0 (full right), 1.0 is center
        wheel_position = (steering - 1.0) * 30  # -30 to +30 degrees

        # Draw speedometer
        self.draw_speedometer(frame, speed)

        # Draw wheel position
        self.draw_wheel_position(frame, wheel_position)

        return frame

    def draw_speedometer(self, frame, speed):
        """Draw a semicircular speedometer on the frame."""
        # Define position and size
        center_x, center_y = 100, 650  # Moved down to y=650
        radius = 100

        # Draw outer semicircle (180 to 360 degrees)
        cv2.ellipse(frame, (center_x, center_y), (radius, radius), 0, 180, 360, (255, 255, 255), 2)

        # Draw ticks and labels
        for i in range(0, 61, 10):
            angle_deg = 180 + (180 * (i / 60.0))  # 180 to 360 degrees
            angle_rad = np.deg2rad(angle_deg)
            tick_length = 10 if i % 20 else 20
            start_x = int(center_x + (radius - tick_length) * np.cos(angle_rad))
            start_y = int(center_y + (radius - tick_length) * np.sin(angle_rad))
            end_x = int(center_x + radius * np.cos(angle_rad))
            end_y = int(center_y + radius * np.sin(angle_rad))
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

            # Put speed labels
            label = f"{i}"
            label_x = int(center_x + (radius - 40) * np.cos(angle_rad))
            label_y = int(center_y + (radius - 40) * np.sin(angle_rad))
            cv2.putText(frame, label, (label_x - 10, label_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw the needle
        angle_deg = 180 + (180 * (speed / 60.0))  # Map speed to angle
        angle_rad = np.deg2rad(angle_deg)
        needle_length = radius - 30
        needle_x = int(center_x + needle_length * np.cos(angle_rad))
        needle_y = int(center_y + needle_length * np.sin(angle_rad))
        cv2.line(frame, (center_x, center_y), (needle_x, needle_y), (0, 0, 255), 3)

        # Put speed text
        cv2.putText(frame, f"Speed: {int(speed)} km/h", (center_x - 70, center_y - 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def draw_wheel_position(self, frame, wheel_position):
        """Draw a wheel position indicator on the frame."""
        # Define position and size
        start_x, start_y = 300, 700  # Lowered position near bottom-center
        end_x, end_y = 300, 600  # Vertical line upwards from the new start

        # Draw base line
        cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

        # Draw wheel position arrow
        arrow_length = 50
        angle_rad = np.deg2rad(wheel_position)
        arrow_x = int(end_x + arrow_length * np.sin(angle_rad))
        arrow_y = int(end_y - arrow_length * np.cos(angle_rad))
        cv2.arrowedLine(frame, (end_x, end_y), (arrow_x, arrow_y), (0, 255, 0), 3, tipLength=0.3)

        # Put wheel position text
        # Replace the degree symbol with 'deg' to avoid rendering issues
        cv2.putText(frame, f"Wheel: {int(wheel_position)} deg", (start_x - 80, end_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
