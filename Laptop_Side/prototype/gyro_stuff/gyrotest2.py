import serial
import time
import math
import threading
from PIL import Image, ImageTk
import tkinter as tk

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

# Global variable to hold the roll angle for GUI
roll_angle = 0.0

# Lock for thread-safe access to roll_angle
roll_lock = threading.Lock()

def reset_mpu6050():
    """Send a killswitch command to reset MPU-6050."""
    ser.write(b'`')
    print("Killswitch activated. Resetting MPU-6050...")
    time.sleep(1)
    ser.reset_input_buffer()  # Clear the serial buffer to start fresh

def process_data(data_line):
    """Parse serial data and calculate roll."""
    global roll, last_time, roll_angle
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

        # Update the global roll_angle variable safely
        with roll_lock:
            roll_angle = roll_degrees

        # Print orientation (optional)
        # if roll_degrees > 30.0:
        #     orientation = "Looking Left"
        # elif roll_degrees < -30.0:
        #     orientation = "Looking Right"
        # else:
        #     orientation = "Looking Forward"

        print(f"Roll Angle: {roll_degrees:.2f} degrees")
        last_time = current_time

    except ValueError as e:
        print(f"Data conversion error: {e}, Data received: {data_line}")

def data_collection_thread():
    """Thread function for collecting MPU-6050 data."""
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
                # print(f"Raw data: {line}")  # Debug: show raw serial data
                if line:
                    process_data(line)

    except Exception as e:
        print(f"Data collection error: {e}")
    finally:
        if ser.is_open:
            ser.close()

def main():
    """Main function to start data collection and GUI."""
    # Start the data collection in a separate thread
    data_thread = threading.Thread(target=data_collection_thread, daemon=True)
    data_thread.start()

    # Set up the GUI in the main thread
    root = tk.Tk()
    root.title("Panoramic Image Viewer")

    # Load the panoramic image
    image_path = "C:/Users/samir/Downloads/1280x480.jpg"  # Replace with your image file
    panoramic_image = Image.open(image_path)

    # Ensure the image is 1280x480
    if panoramic_image.size != (1280, 480):
        panoramic_image = panoramic_image.resize((1280, 480), Image.ANTIALIAS)

    # Create a canvas to display the image
    canvas_width = 640
    canvas_height = 480
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
    canvas.pack()

    # Create the initial image on the canvas
    initial_x_offset = (1280 - 640) // 2  # Start at the center
    box = (initial_x_offset, 0, initial_x_offset + 640, 480)
    cropped_image = panoramic_image.crop(box)
    photo = ImageTk.PhotoImage(cropped_image)
    image_on_canvas = canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo  # Keep a reference to prevent garbage collection

    # Function to update the displayed image based on roll_angle
    def update_image():
        with roll_lock:
            angle = roll_angle  # Get the current roll angle

        # Clamp the roll angle to -90 to +90 degrees
        if angle < -90:
            angle = -90
        elif angle > 90:
            angle = 90

        # Map the roll angle (-90 to +90 degrees) to x-offset (640 to 0 pixels)
        # Leftmost position at angle = +90 degrees (x_offset = 0)
        # Center position at angle = 0 degrees (x_offset = 320)
        # Rightmost position at angle = -90 degrees (x_offset = 640)
        x_offset = int(((-angle + 90) / 180) * (1280 - 640))

        # Ensure x_offset is within bounds
        if x_offset < 0:
            x_offset = 0
        elif x_offset > (1280 - 640):
            x_offset = (1280 - 640)

        # Crop the image to the current view
        box = (x_offset, 0, x_offset + 640, 480)
        cropped_image = panoramic_image.crop(box)

        # Convert the image to PhotoImage
        photo = ImageTk.PhotoImage(cropped_image)

        # Update the image on the canvas
        canvas.itemconfig(image_on_canvas, image=photo)
        canvas.image = photo  # Update the reference

        # Schedule the next update
        root.after(50, update_image)  # Update every 50 milliseconds

    # Start the image update loop
    update_image()

    # Start the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    main()
