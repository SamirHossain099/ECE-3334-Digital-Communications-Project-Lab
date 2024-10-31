import asyncio
import websockets
import pygame
import sys

class Control:
    def __init__(self):
        pygame.init()
        if pygame.joystick.get_count() == 0:
            print("No joystick connected!")
            sys.exit()

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"Connected to joystick: {self.joystick.get_name()}")
        self.uri = "ws://192.168.137.30:8765" # Jetson IP address
        self.running = True

    async def send_joystick_data(self):
        async with websockets.connect(self.uri) as websocket:
            while self.running:
                pygame.event.pump()
                
                axis_value = [self.joystick.get_axis(i) + 1 for i in range(self.joystick.get_numaxes())]
                axis_value_str = ','.join([f"{value:.4f}" for value in axis_value])
                
                await websocket.send(axis_value_str)
                print(f"Sent: {axis_value_str}")

                response = await websocket.recv()
                print(f"Received from server: {response}")

                sys.stdout.flush()
                pygame.time.wait(20)

                keys = pygame.key.get_pressed()
                if keys[pygame.K_q]:
                    print("Exiting due to 'q' key press...")
                    self.running = False

    def run(self):
        asyncio.run(self.send_joystick_data())