"""FastAPI + websocket server for the training monitor.

Runs in a daemon thread inside the trainer process so it can read the bus with
no IPC. The socket pushes at a fixed frame rate and always sends the *latest*
state, never a queue — if the browser stalls, it catches up by skipping.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .bus import BUS

WEB = Path(__file__).parent / "web"


def make_app() -> FastAPI:
    app = FastAPI(title="flypv monitor")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/static")
    def static_payload():
        return JSONResponse(BUS.static())

    @app.get("/api/series")
    def series(stride: int = 1):
        return JSONResponse(BUS.series(stride=stride))

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        await sock.send_json({"kind": "static", **BUS.static()})
        last = -1
        try:
            while True:
                snap = BUS.snapshot()
                if snap["seq"] != last:
                    last = snap["seq"]
                    await sock.send_json({"kind": "frame", **snap})
                await asyncio.sleep(1 / 20)
        except (WebSocketDisconnect, RuntimeError):
            return

    if WEB.exists():
        app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


def serve(host: str = "127.0.0.1", port: int = 8777, log_level: str = "warning"):
    import uvicorn
    uvicorn.run(make_app(), host=host, port=port, log_level=log_level)


def serve_background(host: str = "127.0.0.1", port: int = 8777) -> threading.Thread:
    t = threading.Thread(target=serve, args=(host, port), daemon=True)
    t.start()
    print(f"[flypv] monitor -> http://{host}:{port}")
    return t
