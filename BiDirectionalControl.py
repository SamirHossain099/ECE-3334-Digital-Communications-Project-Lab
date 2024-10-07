import asyncio
import websockets
import sys

async def send_message():
    uri = "ws://192.168.137.30:8765"

    async with websockets.connect(uri) as websocket:
        while True:
            message = input("Enter your message (type 'exit' to quit): ")
            if message.lower() == 'exit':
                print("Exiting...")
                break
            await websocket.send(message)
            print(f"Sent: {message}")

            response = await websocket.recv()
            print(f"Received from server: {response}")
            sys.stdout.flush()

asyncio.run(send_message())
