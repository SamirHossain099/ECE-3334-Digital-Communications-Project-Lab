import pygame
import sys
import asyncio
import websockets

pygame.init()
pygame.joystick.init()

# Function to handle joystick input
def get_joystick_data():
    joystick_data = {}
    
    if pygame.joystick.get_count() == 0:
        print("No joystick connected!")
        return None
    else:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()

        for i in range(joystick.get_numaxes()):
            axis_value = joystick.get_axis(i)
            joystick_data[f"axis_{i}"] = axis_value
        return joystick_data

async def send_joystick_data(uri):
    async with websockets.connect(uri) as websocket:
        while True:
            pygame.event.pump()  # Update Pygame events
            joystick_data = get_joystick_data()
            
            if joystick_data:
                message = str(joystick_data)
                await websocket.send(message)
                print(f"Sent: {message}")
                
                response = await websocket.recv()
                print(f"Received from server: {response}")
            
            await asyncio.sleep(1)  # 1-second delay between sends

async def main():
    uri = "ws://192.168.137.30:8765" 
    await send_joystick_data(uri)

asyncio.run(main())

pygame.quit()
