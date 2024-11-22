# shared_data.py

import threading

# Shared variable for the roll angle
roll_angle = 0.0

# Lock for thread-safe access to roll_angle
roll_lock = threading.Lock()
