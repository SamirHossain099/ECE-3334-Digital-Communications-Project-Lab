import pygame
import sys
import socket

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
            
            axis_value = [self.joystick.get_axis(i)+1 for i in range(self.joystick.get_numaxes())]
            axis_value_str = ','.join([f"{value:.4f}" for value in axis_value])
            
            self.sock.sendto(axis_value_str.encode(), self.server_address)
            print(f"Sent: {axis_value_str}")
            
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