"""FastAPI + websocket server for the training monitor.

The monitor is intentionally lossy: clients receive the newest state available,
never a queue. Live updates are sent as channel deltas so a cheap flight-pose
update does not resend a large unchanged neural-activity frame.
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
WS_HZ = 30.0


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
        last = 0
        try:
            while True:
                snap = BUS.snapshot(since=last)
                if snap["seq"] != last:
                    last = snap["seq"]
                    await sock.send_json({"kind": "frame", **snap})
                await asyncio.sleep(1.0 / WS_HZ)
        except (WebSocketDisconnect, RuntimeError):
            return

    if WEB.exists():
        app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


def serve(host: str = "127.0.0.1", port: int = 8777, log_level: str = "warning"):
    import uvicorn
    uvicorn.run(make_app(), host=host, port=port, log_level=log_level)


def serve_background(host: str = "127.0.0.1", port: int = 8777) -> threading.Thread:
    thread = threading.Thread(target=serve, args=(host, port), daemon=True)
    thread.start()
    print(f"[flypv] monitor -> http://{host}:{port}")
    return thread
