import serial
import time
import threading
from PIL import Image, ImageTk
import tkinter as tk

# Serial port settings (adjust the port name as needed)
SERIAL_PORT = 'COM3'  # Replace with your Arduino's serial port
BAUD_RATE = 115200

# Initialize serial connection
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    ser.reset_input_buffer()  # Clear buffer after opening
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit(1)

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
    """Parse serial data and extract roll angle."""
    global roll_angle

    try:
        # Extract the roll angle from the data_line
        if 'Roll Angle:' in data_line:
            # Split the line to extract the angle
            parts = data_line.split('Roll Angle:')
            if len(parts) == 2:
                angle_part = parts[1].split('degrees')[0].strip()
                roll_degrees = float(angle_part)
                with roll_lock:
                    roll_angle = roll_degrees
                print(f"Processed Roll Angle: {roll_degrees:.2f} degrees")  # Debug
            else:
                print(f"Unexpected data format: {data_line}")
        else:
            print(f"Unexpected data format: {data_line}")

    except ValueError as e:
        print(f"Data conversion error: {e}, Data received: {data_line}")

def data_collection_thread():
    """Thread function for collecting MPU-6050 data."""
    print("Starting data collection in 5 seconds...")
    time.sleep(5)
    print("Data collection started.")

    try:
        while True:
            # Read data from MPU-6050
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"Raw data: {line}")  # Debug: show raw serial data
                if line:
                    process_data(line)
            else:
                time.sleep(0.01)  # Small delay to prevent 100% CPU usage

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
    try:
        panoramic_image = Image.open(image_path)
    except FileNotFoundError:
        print(f"Image file not found: {image_path}")
        exit(1)

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
        angle = max(min(angle, 90), -90)

        # Map the roll angle (-90 to +90 degrees) to x-offset (0 to 640 pixels)
        # Leftmost position at angle = +90 degrees (x_offset = 0)
        # Center position at angle = 0 degrees (x_offset = 320)
        # Rightmost position at angle = -90 degrees (x_offset = 640)
        x_offset = int(((-angle + 90) / 180) * (1280 - 640))

        # Ensure x_offset is within bounds
        if x_offset < 0:
            x_offset = 0
        elif x_offset > (1280 - 640):
            x_offset = (1280 - 640)

        # Debug: Print roll_angle and x_offset
        print(f"Update Image - Roll Angle: {angle:.2f}, X Offset: {x_offset}")  # Debug

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
