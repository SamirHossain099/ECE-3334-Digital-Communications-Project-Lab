# shared_data.py

import threading

# Shared variable for the roll angle
roll_angle = 0.0
wheel_angle = 1.0
throttle_pos = 1.0

# Lock for thread-safe access to roll_angle
roll_lock = threading.Lock()
wheel_angle = threading.Lock()
throttle_pos = threading.Lock()