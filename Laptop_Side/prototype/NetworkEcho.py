import asyncio
import websockets
import sys
#client
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


#server
async def echo(websocket, path):
    print("WebSocket server started on ws://0.0.0.0:8765")
    while True:
        try: 
            message = await websocket.recv()
            print(f"Received: {Message}")
            await websocket.send(f"Echo: {message}")
            sys.stdout.flush()
        except websockets.ConnectionClosed:
            print("Connection Closed")
            break
start_server = websockets.serve(echo, "0.0.0.0",8765)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()