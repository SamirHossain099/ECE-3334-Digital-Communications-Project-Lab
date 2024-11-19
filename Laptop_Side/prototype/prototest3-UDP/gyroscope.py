# gyroscope.py

import serial
import time
import threading
import shared_data  # Import the shared data module

class Gyroscope:
    def __init__(self, serial_port='COM3', baud_rate=115200):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.ser = None
        self.running = True

        # Variables for threshold-based movement detection
        self.previous_roll_angle = None
        self.MIN_CHANGE_THRESHOLD = 0.5  # Minimum change in degrees to consider as movement

        # Variables for reset logic
        self.last_movement_time = time.time()
        self.RESET_DURATION = 3.0  # Time in seconds to wait before resetting
        self.reset_performed = False

    def reset_mpu6050(self):
        """Send a killswitch command to reset MPU-6050."""
        if self.ser and self.ser.is_open:
            self.ser.write(b'`')
            print("Killswitch activated. Resetting MPU-6050...")
            time.sleep(1)
            self.ser.reset_input_buffer()  # Clear the serial buffer to start fresh
        else:
            print("Serial port is not open. Cannot reset MPU-6050.")

    def process_data(self, data_line):
        """Parse serial data and update roll angle based on significant changes."""
        try:
            # Extract the roll angle from the data_line
            if 'Roll Angle:' in data_line:
                # Split the line to extract the angle
                parts = data_line.split('Roll Angle:')
                if len(parts) == 2:
                    angle_part = parts[1].split('degrees')[0].strip()
                    current_roll_angle = float(angle_part)

                    # Initialize previous_roll_angle if it's None
                    if self.previous_roll_angle is None:
                        self.previous_roll_angle = current_roll_angle
                        self.last_movement_time = time.time()  # Initialize the last movement time
                        self.reset_performed = False

                    # Calculate the change in roll angle
                    delta_angle = current_roll_angle - self.previous_roll_angle

                    # Check if the change exceeds the threshold
                    if abs(delta_angle) >= self.MIN_CHANGE_THRESHOLD:
                        with shared_data.roll_lock:
                            shared_data.roll_angle = current_roll_angle
                        print(f"Significant movement detected. Updated Roll Angle: {current_roll_angle:.2f} degrees")  # Debug
                        self.last_movement_time = time.time()  # Update the last movement time
                        self.reset_performed = False  # Allow future resets
                    else:
                        # Ignore minor changes to discard drift
                        print(f"Minor change detected ({delta_angle:.2f} degrees). Ignoring drift.")  # Debug

                        # Check if it's time to reset
                        if (time.time() - self.last_movement_time >= self.RESET_DURATION) and not self.reset_performed:
                            print(f"No significant movement for {self.RESET_DURATION} seconds. Resetting chip and recentering.")
                            self.reset_mpu6050()
                            with shared_data.roll_lock:
                                shared_data.roll_angle = 0.0  # Recenter the view
                            self.reset_performed = True  # Prevent multiple resets until next movement

                    # Update previous_roll_angle
                    self.previous_roll_angle = current_roll_angle
                else:
                    print(f"Unexpected data format: {data_line}")
            else:
                print(f"Unexpected data format: {data_line}")

        except ValueError as e:
            print(f"Data conversion error: {e}, Data received: {data_line}")

    def data_collection_thread(self):
        """Thread function for collecting MPU-6050 data."""
        print("Starting gyroscope data collection in 5 seconds...")
        time.sleep(5)
        print("Gyroscope data collection started.")

        try:
            while self.running:
                # Read data from MPU-6050
                if self.ser and self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    print(f"Raw data: {line}")  # Debug: show raw serial data
                    if line:
                        self.process_data(line)
                else:
                    time.sleep(0.01)  # Small delay to prevent 100% CPU usage

        except serial.SerialException as e:
            print(f"Serial exception: {e}")
            self.running = False
        except Exception as e:
            print(f"Data collection error: {e}")
            self.running = False
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()

    def RunGyro(self):
        """Initialize serial connection and start data collection."""
        while self.running:
            try:
                self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
                self.ser.reset_input_buffer()  # Clear buffer after opening
                self.data_collection_thread()
            except serial.SerialException as e:
                print(f"Error opening serial port: {e}")
                print("Retrying in 5 seconds...")
                time.sleep(5)  # Wait before retrying
            except Exception as e:
                print(f"Unexpected error: {e}")
                self.running = False
