import asyncio
import websockets

async def echo(websocket, path):
    async for message in websocket:
        await websocket.send(message)  # Echoes back received message

# Start the WebSocket server
async def main():
    async with websockets.serve(echo, "localhost", 8765):
        print("WebSocket server running on ws://localhost:8765")
        await asyncio.Future()  # Run forever

asyncio.run(main())
