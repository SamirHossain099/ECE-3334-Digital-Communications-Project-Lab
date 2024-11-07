import sys
import busio
import board
import time
from adafruit_pca9685 import PCA9685
import math
import socket

class Controls:
    def __init__(self):
        # I2C1 on Orin
        self.i2c_bus = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(self.i2c_bus)
        self.pca.frequency = 46
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', 8765))  # Correct port number

    def pulse_width_to_duty_cycle(self, pulse_width_ms, frequency=50):
        pulse_length = 1000000 / frequency  # Pulse length in microseconds
        duty_cycle = int((pulse_width_ms * 1000) / pulse_length * 0xFFFF)
        return duty_cycle

    def receive_joystick_data(self):
        print("Server is running...")
        while True:
            message, address = self.sock.recvfrom(1024)  # Buffer size
            message = message.decode()
            Axis1 = message[0:6]
            Axis2 = message[7:13]
            Axis3 = message[14:20]
            Axis1 = float(Axis1)
            Axis2 = float(Axis2)
            Axis3 = float(Axis3)

            # Clamping and mapping for Axis1
            if Axis1 > 1.5:
                Axis1 = 1.5
            elif Axis1 < 0.5:
                Axis1 = 0.5

            # Map Axis1 from [0.5, 1.5] to [1, 2]
            pulsewidth = 0.0
            if Axis1 > 1.0:
                pulsewidth = 1.5 - (Axis1 - 1.0)
            elif Axis1 < 1.0:
                pulsewidth = 1.5 + (1.0 - Axis1)

            duty_cycle = self.pulse_width_to_duty_cycle(pulsewidth)
            self.pca.channels[0].duty_cycle = duty_cycle

            scale = math.exp(2) - 1
            if Axis2 < 1.0:
                Axis2 = 1.0
            if Axis3 < 1.0:
                Axis3 = 1.0
            axis2_contrib = ((math.exp(abs(float(Axis2) - 2)) - 1) / scale)
            axis3_contrib = ((math.exp(abs(float(Axis3) - 2)) - 1) / scale)
            pulsewidth = 1.5 + 0.5 * axis2_contrib - 0.5 * axis3_contrib

            duty_cycle = self.pulse_width_to_duty_cycle(pulsewidth)
            self.pca.channels[1].duty_cycle = duty_cycle

            print(f"Axis1: {Axis1} Axis2: {Axis2} Axis3: {Axis3}")

            sys.stdout.flush()

    def run(self):
        self.receive_joystick_data()

if __name__ == "__main__":
    controls = Controls()
    controls.run()