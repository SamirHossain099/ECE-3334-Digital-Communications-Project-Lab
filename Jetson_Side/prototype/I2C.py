# import busio
# import board
# import adafruit_servokit as ServoKit
# from adafruit_pca9685 import PCA9685
# import time

# i2c_bus = busio.I2C(board.SCL, board.SDA)
# pca = PCA9685(i2c_bus)
# pca.frequency = 46.5
# pca.channels[0].duty_cycle = 0x7FFF
# time.sleep(1000)

import busio
import board
import time
from adafruit_pca9685 import PCA9685

# Set up I2C and PCA9685
i2c_bus = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c_bus)
pca.frequency = 46  # 50Hz for servo control

# Helper function to convert pulse width to duty cycle
def pulse_width_to_duty_cycle(pulse_width_ms, frequency=50):
    pulse_length = 1000000 / frequency  # Pulse length in microseconds
    duty_cycle = int((pulse_width_ms * 1000) / pulse_length * 0xFFFF)
    return duty_cycle

# Set pulse widths
pulse_widths = [1.0, 1.5, 2.0]  # 1ms, 1.5ms, 2ms pulse widths
duration = 10  # 10 seconds for each pulse width

for pulse_width in pulse_widths:
    duty_cycle = pulse_width_to_duty_cycle(pulse_width)
    pca.channels[0].duty_cycle = duty_cycle
    print(f"Setting pulse width to {pulse_width}ms")
    time.sleep(duration)

# Set pulse width to 0ms (effectively turning off the servo)
pca.channels[0].duty_cycle = 0
print("Setting pulse width to 0ms, turning off the servo")
pca.deinit()
