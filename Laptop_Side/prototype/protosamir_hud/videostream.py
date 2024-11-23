# videostream.py

import cv2
import glob
import os
import time
import numpy as np
import shared_data  # Import shared_data to access roll_angle, steering, throttle, and brake

class VideoStream:
    def __init__(self, camera1_folder="D:/Lab/Terminal1/", camera2_folder="D:/Lab/Terminal2/"):
    # def __init__(self, camera1_folder="D:/temp/camera1/", camera2_folder="D:/temp/camera2/"): #Deuce
        self.camera1_folder = camera1_folder
        self.camera2_folder = camera2_folder
        self.running = True
        self.current_speed = 0.0  # Initialize current speed

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
        # Define speed parameters
        max_speed = 60.0  # Maximum speed in km/h
        damping_factor = 0.1  # Smoothing factor for speed transitions

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

            cropped_frame_resize = cv2.resize(cropped_frame, (1920, 1080))  # Adjust as needed

            # Overlay HUD on the cropped_frame_resize
            hud_frame = self.add_hud(cropped_frame_resize, max_speed, damping_factor)

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

    def add_hud(self, frame, max_speed, damping_factor):
        """Add HUD elements to the frame."""
        # Retrieve steering, throttle, and brake values safely
        with shared_data.steering_lock:
            steering = shared_data.steering_value
        with shared_data.throttle_lock:
            throttle = shared_data.throttle_value
        with shared_data.brake_lock:
            brake = shared_data.brake_value

        # Map throttle and brake to speed change
        # Throttle and brake both range from 0.0 to 2.0, where lower throttle means higher speed
        # Lower brake means less braking
        # Compute desired speed change: throttle_effect - brake_effect
        throttle_effect = (2.0 - throttle)  # 0.0 (max throttle) to 2.0 (no throttle)
        brake_effect = (2.0 - brake)        # 0.0 (max brake) to 2.0 (no brake)

        # Calculate net speed effect
        net_speed_effect = throttle_effect - brake_effect  # Range: -2.0 to +2.0

        # Update current_speed with smoothing
        self.current_speed += (net_speed_effect * max_speed / 2.0 - self.current_speed) * damping_factor

        # Clamp current_speed to -max_speed to +max_speed
        self.current_speed = max(min(self.current_speed, max_speed), -max_speed)

        # Draw speed meter
        self.draw_speed_meter(frame, self.current_speed, max_speed)

        # Draw wheel position
        self.draw_wheel_position(frame, steering)

        return frame

    def draw_speed_meter(self, frame, speed, max_speed):
        """Draw a semicircular tachometer-like speed meter on the frame."""
        # Define position and size
        center_x, center_y = 100, 700  # Positioned near bottom-left corner
        radius = 100

        # Draw outer semicircle (180 to 360 degrees)
        cv2.ellipse(frame, (center_x, center_y), (radius, radius), 0, 180, 360, (255, 255, 255), 2)

        # Draw the needle based on speed
        # Map speed to angle (180 to 360 degrees)
        # speed ranges from -max_speed to +max_speed
        # speed=0 -> 270 degrees (up)
        # speed=+max_speed -> 360 degrees (right)
        # speed=-max_speed -> 180 degrees (left)
        angle_deg = 270 + (speed / max_speed) * 90  # 180 to 360 degrees
        angle_deg = max(min(angle_deg, 360), 180)   # Clamp to 180-360
        angle_rad = np.deg2rad(angle_deg)
        needle_length = radius - 30
        needle_x = int(center_x + needle_length * np.cos(angle_rad))
        needle_y = int(center_y + needle_length * np.sin(angle_rad))
        cv2.line(frame, (center_x, center_y), (needle_x, needle_y), (0, 0, 255), 3)

        # Optional: Draw a filled circle at the center
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # Remove numbers from the meter (as per user request)
        # Previously, speed labels were added; now they are omitted

    def draw_wheel_position(self, frame, steering):
        """Draw a wheel position indicator on the frame."""
        # Define position and size
        start_x, start_y = 300, 700  # Lowered position near bottom-center
        end_x, end_y = 300, 600      # Vertical line upwards from the new start

        # Remove the white base line by commenting out the following line
        # cv2.line(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

        # Calculate wheel position angle
        # steering ranges from 0.0 (full left) to 2.0 (full right), 1.0 is center
        # Map steering to angle: -30 to +30 degrees
        wheel_position = (steering - 1.0) * 30  # -30 to +30 degrees
        wheel_position = max(min(wheel_position, 30), -30)  # Clamp to -30 to +30

        # Draw wheel position arrow
        arrow_length = 50
        angle_rad = np.deg2rad(wheel_position)
        arrow_x = int(end_x + arrow_length * np.sin(angle_rad))
        arrow_y = int(end_y - arrow_length * np.cos(angle_rad))
        cv2.arrowedLine(frame, (end_x, end_y), (arrow_x, arrow_y), (0, 255, 0), 3, tipLength=0.3)

        # Put wheel position text
        # Remove the degree symbol by updating the text format
        cv2.putText(frame, f"Wheel: {int(wheel_position)}", (start_x - 80, end_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
