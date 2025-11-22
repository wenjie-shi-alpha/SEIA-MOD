"""FastAPI WebSocket binary streaming server."""
from __future__ import annotations

import asyncio
import struct
import time
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, List, Sequence

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

MAGIC = b"SEIA"
HEADER_STRUCT = struct.Struct("<4sHIdHI")
VERSION = 1


@dataclass
class AgentStreamConfig:
    """Configuration for the binary streaming channel."""

    max_queue: int = 64
    components: int = 3  # x, y, damage
    dtype: np.dtype = np.dtype("<f4")


class AgentStateBroadcaster:
    """Stores outgoing frames and serves them to WebSocket subscribers."""

    def __init__(self, config: AgentStreamConfig) -> None:
        self.config = config
        self._queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=config.max_queue)

    async def publish_frame(self, frame: bytes) -> None:
        """Publish a raw binary frame to all listeners."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(frame)

    async def publish_agents(
        self,
        agent_positions: np.ndarray,
        damage_rates: np.ndarray,
        agent_ids: Sequence[str] | None = None,
        timestamp: float | None = None,
    ) -> bytes:
        """Encode agent state arrays and publish them."""
        timestamp = timestamp or time.time()
        frame = encode_agent_frame(
            agent_positions.astype(self.config.dtype),
            damage_rates.astype(self.config.dtype),
            agent_ids=agent_ids,
            timestamp=timestamp,
            components=self.config.components,
        )
        await self.publish_frame(frame)
        return frame

    async def stream(self) -> AsyncIterator[bytes]:
        """Yield frames as they arrive."""
        while True:
            payload = await self._queue.get()
            yield payload


def encode_agent_frame(
    positions: np.ndarray,
    damage_rates: np.ndarray,
    agent_ids: Sequence[str] | None,
    timestamp: float,
    components: int,
) -> bytes:
    """Return a binary frame representing agent states."""
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions array must be of shape (N, 2)")
    if damage_rates.shape[0] != positions.shape[0]:
        raise ValueError("damage_rates must align with positions")
    damage_rates = damage_rates.reshape(-1, 1)
    payload_matrix = np.concatenate([positions, damage_rates], axis=1).astype("<f4")
    agent_count = payload_matrix.shape[0]
    id_list = list(agent_ids or [])
    if len(id_list) > agent_count:
        id_list = id_list[:agent_count]
    if len(id_list) < agent_count:
        id_list.extend([""] * (agent_count - len(id_list)))
    ids_blob = "\0".join(id_list).encode("utf-8")
    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        agent_count,
        timestamp,
        components,
        len(ids_blob),
    )
    return header + ids_blob + payload_matrix.tobytes()


class AgentState(BaseModel):
    agent_id: str = Field(..., examples=["agent_001"])
    x: float
    y: float
    damage: float = 0.0


class AgentBatch(BaseModel):
    agents: List[AgentState]
    timestamp: float | None = None


app = FastAPI(title="SEIA-Mod Streaming Server")
broadcaster = AgentStateBroadcaster(AgentStreamConfig())


async def agent_state_stream() -> AsyncIterator[bytes]:
    """Yield binary agent state payloads."""
    async for payload in broadcaster.stream():
        yield payload


@app.post("/ingest")
async def ingest_agent_batch(batch: AgentBatch):
    """HTTP ingestion endpoint for feeding agent states into the stream."""
    if not batch.agents:
        return JSONResponse({"status": "ignored", "reason": "empty batch"}, status_code=status.HTTP_200_OK)
    positions = np.array([[agent.x, agent.y] for agent in batch.agents], dtype="<f4")
    damage = np.array([agent.damage for agent in batch.agents], dtype="<f4")
    await broadcaster.publish_agents(
        agent_positions=positions,
        damage_rates=damage,
        agent_ids=[agent.agent_id for agent in batch.agents],
        timestamp=batch.timestamp or time.time(),
    )
    return {"status": "ok", "agents": len(batch.agents)}


@app.websocket("/ws/agents")
async def agents_endpoint(socket: WebSocket) -> None:  # pragma: no cover - network I/O
    await socket.accept()
    try:
        async for payload in agent_state_stream():
            await socket.send_bytes(payload)
    except WebSocketDisconnect:
        return
