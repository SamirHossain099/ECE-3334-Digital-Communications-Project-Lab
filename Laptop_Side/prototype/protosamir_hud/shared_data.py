# shared_data.py

import threading

# Shared variable for the roll angle
roll_angle = 0.0

# Lock for thread-safe access to roll_angle
roll_lock = threading.Lock()

# New Shared Variables for HUD
steering_value = 1.0  # Centered by default (range: 0.0 to 2.0)
throttle_value = 2.0  # No speed by default (range: 0.0 to 2.0)

# Locks for thread-safe access
steering_lock = threading.Lock()
throttle_lock = threading.Lock()
