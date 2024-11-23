# sendcontrols.py

import pygame
import sys
import socket
import shared_data  # Import shared_data to update steering and throttle

class Send_Control_Data:
    def __init__(self):
        pygame.init()
        if pygame.joystick.get_count() == 0:
            print("No joystick connected!")
            sys.exit()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Connected to joystick: {self.joystick.get_name()}")
        self.server_address = ('10.161.189.106', 8765)  # Jetson TTU IP and port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_joystick_data(self):
        running = True
        while running:
            pygame.event.pump()
            
            axis_values = [self.joystick.get_axis(i) + 1 for i in range(self.joystick.get_numaxes())]
            axis_values_str = ','.join([f"{value:.4f}" for value in axis_values])
            
            self.sock.sendto(axis_values_str.encode(), self.server_address)
            # print(f"Sent: {axis_values_str}")
            
            # Update shared_data with steering and throttle
            if len(axis_values) >= 2:
                steering = axis_values[0]  # First value: Steering
                throttle = axis_values[1]  # Second value: Throttle
                
                with shared_data.steering_lock:
                    shared_data.steering_value = steering
                with shared_data.throttle_lock:
                    shared_data.throttle_value = throttle
                # Debug: Print updated values
                # print(f"Updated Steering: {steering}, Throttle: {throttle}")
            
            pygame.time.wait(10)
            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]:
                print("Exiting due to 'q' key press...")
                running = False

    def start_server(self):
        self.send_joystick_data()

if __name__ == "__main__":
    sender = Send_Control_Data()
    sender.start_server()
