
import asyncio
import websockets
import pygame
import sys

pygame.init()
if pygame.joystick.get_count() == 0:
    print("No joystick connected!")
    sys.exit() 

joystick = pygame.joystick.Joystick(0)
joystick.init()
print(f"Connected to joystick: {joystick.get_name()}")

async def send_joystick_data():
    uri = "ws://192.168.137.30:8765"
    
    async with websockets.connect(uri) as websocket:
        running = True
        
        while running:

            pygame.event.pump()  
            
            axis_value = [joystick.get_axis(i)+1 for i in range(joystick.get_numaxes())]
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
                running = False

asyncio.run(send_joystick_data())

