import serial
import time
import math

# Serial port settings (adjust the port name as needed)
SERIAL_PORT = 'COM3'  # Replace with your Arduino's serial port
BAUD_RATE = 115200

# Initialize serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
ser.reset_input_buffer()  # Clear buffer after opening

# Variables for calculating roll
roll = 0.0
last_time = time.time()
last_reset_time = time.time()  # Initialize reset time globally

# Define reset interval in seconds (adjust as needed)
RESET_INTERVAL = 60

def reset_mpu6050():
    """Send a killswitch command to reset MPU-6050."""
    ser.write(b'`')
    print("Killswitch activated. Resetting MPU-6050...")
    time.sleep(1)
    ser.reset_input_buffer()  # Clear the serial buffer to start fresh

def process_data(data_line):
    """Parse serial data and calculate roll."""
    global roll, last_time
    alpha = 0.98  # Complementary filter constant

    try:
        values = [float(x) for x in data_line.strip().split(',')]
        if len(values) != 6:
            print(f"Unexpected data format: {data_line}")
            return

        accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = values
        current_time = time.time()
        delta_time = current_time - last_time

        # Calculate roll from accelerometer data
        accel_roll = math.degrees(math.atan2(accel_y, accel_z))

        # Calculate roll angle from gyro data
        gyro_roll_rate = gyro_x  # Assuming gyro_x is in degrees/sec
        gyro_roll = roll + gyro_roll_rate * delta_time

        # Complementary filter to combine both measurements
        roll = alpha * gyro_roll + (1 - alpha) * accel_roll

        # Normalize roll angle to -180 to 180 degrees
        roll_degrees = (roll + 180) % 360 - 180

        # Determine orientation
        if roll_degrees > 30.0:
            orientation = "Looking Left"
        elif roll_degrees < -30.0:
            orientation = "Looking Right"
        else:
            orientation = "Looking Forward"

        print(f"{orientation}, Roll Angle: {roll_degrees:.2f} degrees")
        last_time = current_time

    except ValueError as e:
        print(f"Data conversion error: {e}, Data received: {data_line}")

def main():
    global roll, last_time, last_reset_time  # Ensure we can modify the global variables
    print("Starting data collection in 5 seconds...")
    time.sleep(5)
    print("Data collection started.")

    try:
        while True:
            # Periodically reset MPU-6050 to avoid drift or errors
            current_time = time.time()
            if current_time - last_reset_time >= RESET_INTERVAL:
                reset_mpu6050()
                roll = 0.0
                last_time = current_time
                last_reset_time = current_time  # Reset the timer

            # Read data from MPU-6050
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"Raw data: {line}")  # Debug: show raw serial data
                if line:
                    process_data(line)

    except KeyboardInterrupt:
        print("Exiting program.")
    finally:
        if ser.is_open:
            ser.close()

if __name__ == "__main__":
    main()
