import time

class FPSCalculator:
    def __init__(self):
        self.prev_time = time.time()
        self.fps_counter = 0

    def calculate_fps(self):
        while True:
            current_time = time.time()
            elapsed_time = current_time - self.prev_time
            if elapsed_time >= 1.0:
                fps = self.fps_counter
                print(f"FPS: {fps}")
                self.fps_counter = 0
                self.prev_time = current_time
            time.sleep(0.1)
