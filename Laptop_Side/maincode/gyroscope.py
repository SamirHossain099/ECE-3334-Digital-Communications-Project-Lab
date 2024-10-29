import serial
import time
import math
import msvcrt

class Gyroscope:
    def __init__(self):
        self.SERIAL_PORT = 'COM3'
        self.BAUD_RATE = 115200
        self.ser = serial.Serial(self.SERIAL_PORT, self.BAUD_RATE, timeout=1)
        self.roll = 0.0
        self.last_time = time.time()
    
    def reset_mpu6050(self):
        self.ser.write(b'`')
        print("Killswitch activated. Resetting MPU-6050...")
        time.sleep(1)
    
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

            if roll_degrees > 30.0:
                orientation = "Looking Left"
            elif roll_degrees < -30.0:
                orientation = "Looking Right"
            else:
                orientation = "Looking Forward"

            print(f"{orientation}, Roll Angle: {roll_degrees:.2f} degrees")

            self.last_time = current_time

        except ValueError:
            pass
        
    def RunGyro():
        try:
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8')
                    process_data(line)

                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'`':
                        reset_mpu6050()
                        self.roll = 0.0
                        self.last_time = time.time()
        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            self.ser.close()

