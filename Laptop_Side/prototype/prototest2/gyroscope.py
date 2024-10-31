# gyroscope.py

import serial
import time
import threading
import shared_data  # Import the shared data module

class Gyroscope:
    def __init__(self, serial_port='COM5', baud_rate=115200):
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.ser = None
        self.running = True

    def process_data(self, data_line):
        """Parse serial data and extract roll angle."""
        try:
            # Extract the roll angle from the data_line
            if 'Roll Angle:' in data_line:
                # Split the line to extract the angle
                parts = data_line.split('Roll Angle:')
                if len(parts) == 2:
                    angle_part = parts[1].split('degrees')[0].strip()
                    roll_degrees = float(angle_part)
                    with shared_data.roll_lock:
                        shared_data.roll_angle = roll_degrees
                    print(f"Processed Roll Angle: {roll_degrees:.2f} degrees")  # Debug
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
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    print(f"Raw data: {line}")  # Debug: show raw serial data
                    if line:
                        self.process_data(line)
                else:
                    time.sleep(0.01)  # Small delay to prevent 100% CPU usage

        except Exception as e:
            print(f"Data collection error: {e}")
        finally:
            if self.ser and self.ser.is_open:
                self.ser.close()

    def RunGyro(self):
        """Initialize serial connection and start data collection."""
        try:
            self.ser = serial.Serial(self.serial_port, self.baud_rate, timeout=1)
            self.ser.reset_input_buffer()  # Clear buffer after opening
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            return

        self.data_collection_thread()
