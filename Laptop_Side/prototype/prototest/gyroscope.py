# gyroscope.py
import serial
import time
import math

class Gyroscope:
    def __init__(self, port='COM3', baud_rate=115200):
        self.ser = serial.Serial(port, baud_rate, timeout=1)
        self.roll = 0.0
        self.last_time = time.time()
        self.orientation = "Looking Forward"

    def reset_mpu6050(self):
        self.ser.write(b'`')
        time.sleep(1)
        self.roll = 0.0
        self.last_time = time.time()

    def process_data(self, data_line):
        try:
            values = [float(x) for x in data_line.strip().split(',')]
            if len(values) != 6:
                return

            accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = values
            current_time = time.time()
            delta_time = current_time - self.last_time

            gyro_x_rad = math.radians(gyro_x)
            self.roll += gyro_x_rad * delta_time
            roll_degrees = math.degrees(self.roll)

            if roll_degrees > 180.0:
                roll_degrees -= 360.0
            elif roll_degrees < -180.0:
                roll_degrees += 360.0

            # Determine orientation
            if roll_degrees > 30.0:
                self.orientation = "Looking Left"
            elif roll_degrees < -30.0:
                self.orientation = "Looking Right"
            else:
                self.orientation = "Looking Forward"

            self.last_time = current_time

        except ValueError:
            pass

    def RunGyro(self):
        while True:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8')
                self.process_data(line)

    def get_orientation(self):
        return self.orientation
