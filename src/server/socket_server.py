"""FastAPI WebSocket binary streaming server placeholder."""
from __future__ import annotations

from typing import AsyncIterator

from fastapi import FastAPI, WebSocket

app = FastAPI(title="SEIA-Mod Streaming Server")


async def agent_state_stream() -> AsyncIterator[bytes]:
    """Yield binary agent state payloads."""
    yield b""  # TODO: stream from simulation loop


@app.websocket("/ws/agents")
async def agents_endpoint(socket: WebSocket) -> None:  # pragma: no cover - I/O heavy stub
    await socket.accept()
    async for payload in agent_state_stream():
        await socket.send_bytes(payload)
