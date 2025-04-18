import os
import threading
import asyncio

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Define a global event loop that will be shared
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


# Create a connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Error sending message: {e}")
                # Don't remove here, we'll let the disconnect handler manage this


app = FastAPI()
manager = ConnectionManager()

# Create an assistant placeholder
assistant = None


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Python: received from client: {data}")
            if data == "run_query":
                await run_query()
            elif data == "check_assistant_running":
                result = check_assistant_running()
                await websocket.send_text(result)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Python: client disconnected")


async def run_query():
    global assistant
    print("Python: assistant listening")
    if assistant and hasattr(assistant, 'start_recording'):
        # Use asyncio.to_thread for potentially blocking operations
        await loop.run_in_executor(None, assistant.start_recording)
        return {"status": "Recording started via run_query"}
    else:
        print("Warning: Assistant not initialized yet")
        return {"status": "Assistant not ready"}


def check_assistant_running():
    global assistant
    if assistant and getattr(assistant, 'ready', False):
        return "NeonAssistant initialized"
    else:
        return "false"


async def initialize_assistant():
    global assistant, manager, loop
    print("Initializing NeonAssistant...")

    # Import here instead of at the top to avoid circular imports
    from neon_assistant import NeonAssistant
    # Create assistant instance
    assistant = NeonAssistant(manager, loop)
    print("NeonAssistant initialization complete")

    # Broadcast initialization completed
    await manager.broadcast("NeonAssistant initialized")
    return assistant


# Run initialization in the background
def start_assistant_init():
    global loop
    asyncio.run_coroutine_threadsafe(initialize_assistant(), loop)


# Function to start the server with the shared event loop
def run_server():
    global loop
    config = uvicorn.Config(app, host='127.0.0.1', port=5372, loop=loop)
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())


if __name__ == '__main__':
    # Start the server in a thread
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Await 2 secs
    loop.call_later(2, start_assistant_init)

    # Keep the main thread alive
    try:
        server_thread.join()
    except KeyboardInterrupt:
        print("Server shutting down...")
        loop.stop()
