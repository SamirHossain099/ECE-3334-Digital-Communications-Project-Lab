import asyncio
import websockets
import sys
import busio
import board
import time
from adafruit_pca9685 import PCA9685
import math
# #echo example
# async def echo(websocket, path):
#     print("WebSocket server started on ws://0.0.0.0:8765")
#     while True:
#         try:
#             message = await websocket.recv()
#             print(f"Received: {message}")
#             await websocket.send(f"Echo: {message}")
#             sys.stdout.flush()
#         except websockets.ConnectionClosed:
#             print("Connection Closed")
#             break   

# start_server = websockets.serve(echo, "0.0.0.0", 8765)
# asyncio.get_event_loop().run_until_complete(start_server)
# asyncio.get_event_loop().run_forever()

# Joystick data stream
# async def receive_joystick_data(websocket, path):
#     print("Client Connected")
#     try:
#         while True:
#             message = await websocket.recv()
#             print(f"Received joystick data: {message}")
#             sys.stdout.flush()
#             await websocket.send(f"Echo: {message}")

#     except websockets.ConnectionClosed:
#         print("Client disconnected.")
#         sys.stdout.flush()


# start_server = websockets.serve(receive_joystick_data, "0.0.0.0", 8765)
# asyncio.get_event_loop().run_until_complete(start_server)
# print("Server is running...")
# asyncio.get_event_loop().run_forever()

#I2C1 on orin
i2c_bus = busio.I2C(board.SCL, board.SDA)
pca = PCA9685(i2c_bus)
pca.frequency = 46

def pulse_width_to_duty_cycle(pulse_width_ms, frequency=50):
    pulse_length = 1000000 / frequency  # Pulse length in microseconds
    duty_cycle = int((pulse_width_ms * 1000) / pulse_length * 0xFFFF)
    return duty_cycle

async def receive_joystick_data(websocket, path):
    print("Client Connected")
    try:
        while True:
            message = await websocket.recv()
            Axis1 = message[0:6]
            Axis2 = message[7:13]
            Axis3 = message[14:20]            
            Axis1 = float(Axis1)
            Axis2 = float(Axis2)
            Axis3 = float(Axis3)

            # Clamping and mapping for Axis1
            if Axis1 > 1.5:
                Axis1 = 1.5
            elif Axis1 < 0.5:
                Axis1 = 0.5

            # Map Axis1 from [0.5, 1.5] to [1, 2]
            # pulsewidth = 1 + (Axis1 - 0.5)
            pulsewidth = 0.0
            if Axis1 > 1.0:
                pulsewidth = 1.5 - (Axis1 - 1.0)
            elif Axis1 < 1.0:
                pulsewidth = 1.5 + (1.0 - Axis1)
                 
            duty_cycle = pulse_width_to_duty_cycle(pulsewidth)
            pca.channels[0].duty_cycle = duty_cycle

            scale = math.exp(2) - 1
            if Axis2 < 1.0:
                Axis2 = 1.0
            if Axis3 < 1.0:
                Axis3 = 1.0
            axis2_contrib = ((math.exp(abs(float(Axis2) - 2)) - 1) / scale)
            axis3_contrib = ((math.exp(abs(float(Axis3) - 2)) - 1) / scale)
            pulsewidth = 1.5 + 0.5 * axis2_contrib - 0.5 * axis3_contrib

            duty_cycle = pulse_width_to_duty_cycle(pulsewidth)
            pca.channels[1].duty_cycle = duty_cycle

            print(f"Axis1: {Axis1} Axis2: {Axis2} Axis3: {Axis3}")

            sys.stdout.flush()
            await websocket.send(f"Echo: {message}")

    except websockets.ConnectionClosed:
        print("Client disconnected.")
        sys.stdout.flush()


start_server = websockets.serve(receive_joystick_data, "0.0.0.0", 8765)
asyncio.get_event_loop().run_until_complete(start_server)
print("Server is running...")
asyncio.get_event_loop().run_forever()