import os
import threading

import uvicorn
from flask import Flask, jsonify
import random
import time
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

assistant = None


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
                print("Error sending message", e)


app = FastAPI()
manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            print("Python: received from client:", data)
            match data:
                case "run_query":
                    run_query()
                case "check_assistant_running":
                    await websocket.send_text(check_assistant_running())
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("Python: client disconnected")


# @app.post("/run_query")
def run_query():
    print("Python: assistant listening")
    assistant.start_recording()
    return {"status": "Recording started via run_query"}


# @app.get("/assistant_running")
def check_assistant_running():
    if assistant and assistant.ready:
        return "NeonAssistant initialized"
    else:
        return "false"


def run_server():
    uvicorn.run(app, host='127.0.0.1', port=5372)


if __name__ == '__main__':
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

from main import NeonAssistant


class NeonAssistantWS(NeonAssistant):
    def __init__(self, manager: ConnectionManager, *args, **kwargs):
        self.ready = False
        self.manager = manager
        self.loop = asyncio.get_event_loop()
        super().__init__(*args, **kwargs)
        self.loop.create_task(self.manager.broadcast("NeonAssistant initialized"))
        self.ready = True

    def start_recording(self):
        super().start_recording()
        self.loop.create_task(self.manager.broadcast("Recording started"))

    def cleanup_recording(self):
        self.loop.create_task(self.manager.broadcast("Recording cleaned up"))
        super().cleanup_recording()


assistant = NeonAssistantWS(manager)
print("Ready")
if __name__ == '__main__':
    server_thread.join()
