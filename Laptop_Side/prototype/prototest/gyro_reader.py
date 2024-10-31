# gyro_reader.py
import serial
import time
import math

class GyroReader:
    def __init__(self, serial_port='COM3', baud_rate=115200):
        self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
        self.roll = 0.0
        self.last_time = time.time()
        self.orientation = "Looking Forward"

    def reset_mpu6050(self):
        """Send a killswitch command to reset MPU-6050."""
        self.ser.write(b'`')
        print("Killswitch activated. Resetting MPU-6050...")
        time.sleep(1)
        self.ser.reset_input_buffer()  # Clear the serial buffer

    def get_orientation(self):
        return self.orientation

    def process_data(self, data_line):
        """Parse serial data and calculate roll."""
        try:
            values = [float(x) for x in data_line.strip().split(',')]
            if len(values) != 6:
                return

            accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = values
            current_time = time.time()
            delta_time = current_time - self.last_time

            # Calculate roll angle from gyro data
            self.roll += math.radians(gyro_x) * delta_time
            roll_degrees = math.degrees(self.roll)

            # Normalize roll angle to -180 to 180 degrees
            roll_degrees = (roll_degrees + 180) % 360 - 180

            # Determine orientation
            if roll_degrees > 30.0:
                self.orientation = "Looking Left"
            elif roll_degrees < -30.0:
                self.orientation = "Looking Right"
            else:
                self.orientation = "Looking Forward"

            self.last_time = current_time

        except ValueError:
            print("Data conversion error.")

    def run(self):
        """Continuously read and process gyro data."""
        try:
            while True:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    self.process_data(line)
        except KeyboardInterrupt:
            print("Exiting GyroReader.")
        finally:
            self.ser.close()
