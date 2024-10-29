import serial
import time
import math
import msvcrt

# Serial port settings (adjust the port name as needed)
SERIAL_PORT = 'COM3'  # Replace with your Arduino's serial port
BAUD_RATE = 115200

# Initialize serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

# Variables for calculating roll
roll = 0.0
last_time = time.time()

def reset_mpu6050():
    # Send the killswitch character to the Arduino
    ser.write(b'`')
    print("Killswitch activated. Resetting MPU-6050...")
    time.sleep(1)  # Wait for the Arduino to reset the sensor

def process_data(data_line):
    global roll, last_time

    # Parse the CSV data
    try:
        values = [float(x) for x in data_line.strip().split(',')]
        if len(values) != 6:
            return

        accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = values

        current_time = time.time()
        delta_time = current_time - last_time

        # Gyroscope X-axis data (rotation around X-axis in rad/s)
        gyro_x_rad = math.radians(gyro_x)

        # Integrate gyroscope data over time to get roll angle in radians
        roll += gyro_x_rad * delta_time

        # Convert roll from radians to degrees
        roll_degrees = math.degrees(roll)

        # Normalize roll_degrees to the range -180 to 180 degrees
        if roll_degrees > 180.0:
            roll_degrees -= 360.0
        elif roll_degrees < -180.0:
            roll_degrees += 360.0

        # Determine head orientation based on roll angle
        if roll_degrees > 30.0:
            orientation = "Looking Left"
        elif roll_degrees < -30.0:
            orientation = "Looking Right"
        else:
            orientation = "Looking Forward"

        # Print the orientation and roll angle
        print(f"{orientation}, Roll Angle: {roll_degrees:.2f} degrees")

        last_time = current_time

    except ValueError:
        # Handle any parsing errors
        pass

try:
    while True:
        # Read a line from the serial port
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8')
            process_data(line)

        # Check for user input (killswitch)
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'`':
                reset_mpu6050()
                # Reset roll and last_time
                roll = 0.0
                last_time = time.time()

except KeyboardInterrupt:
    print("Exiting...")
finally:
    ser.close()
