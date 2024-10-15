# import asyncio
# import websockets
# import sys
# import pygame
# import keyboard

# # WebSocket server IP address and port
# ws_uri = "ws://192.168.137.30:8765"

# # Initialize Pygame
# pygame.init()

# if pygame.joystick.get_count() == 0:
#     print("No joystick connected!")
# else:
#     # Get the first joystick
#     joystick = pygame.joystick.Joystick(0)
#     joystick.init()
#     print("Connected to:", joystick.get_name())
#     print("Number of axes:", joystick.get_numaxes())

#     async def send_joystick_data():
#         try:
#             async with websockets.connect(ws_uri) as websocket:
#                 running = True
#                 axis_value = [0] * joystick.get_numaxes()  # Store the axis values

#                 while running:
#                     for event in pygame.event.get():
#                         if event.type == pygame.QUIT:
#                             running = False
                    
#                     for i in range(joystick.get_numaxes()):
#                         axis_value[i] = joystick.get_axis(i)

#                     # Format axis values as a string
#                     axis_value_str = ','.join([f"{value:.2f}" for value in axis_value])

#                     # Send joystick data to WebSocket server
#                     await websocket.send(axis_value_str)
#                     print(f"Sent to server: {axis_value_str}")

#                     # Simulate some delay
#                     pygame.time.wait(1000)

#                     # Check for 'q' key to exit
#                     if keyboard.is_pressed('q'):
#                         print("Exiting...")
#                         break

#         except Exception as e:
#             print(f"WebSocket error: {e}")

#     # Run the asyncio loop for WebSocket communication
#     asyncio.run(send_joystick_data())

# pygame.quit()

import asyncio
import websockets
import pygame
import sys

# Initialize pygame joystick
pygame.init()
if pygame.joystick.get_count() == 0:
    print("No joystick connected!")
    sys.exit()  # Exit if no joystick is found

# Get the first joystick
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
            pygame.time.wait(30)  

            
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q]: 
                print("Exiting due to 'q' key press...")
                running = False

asyncio.run(send_joystick_data())

