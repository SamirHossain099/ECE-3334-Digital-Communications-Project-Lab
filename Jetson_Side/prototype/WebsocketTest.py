import asyncio 
import websockets
import time

async def ping_test(uri):
    async with websockets.connect(uri) as websocket:
        start_time = time.time()
        await websocket.send("ping")
        await websocket.recv()
        latency = time.time() - start_time
        print(f"Latency: {latency * 1000:.2f} ms")

uri = "ws:// :8765"
asyncio.run(ping_test(uri))
