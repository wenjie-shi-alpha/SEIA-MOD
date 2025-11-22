"""Tests for the L4 FastAPI/WebSocket layer."""
from __future__ import annotations

import struct

import numpy as np
from fastapi.testclient import TestClient

from src.server.socket_server import (
    HEADER_STRUCT,
    MAGIC,
    VERSION,
    app,
    encode_agent_frame,
)


def decode_frame(payload: bytes):
    header = HEADER_STRUCT.unpack_from(payload, 0)
    ids_length = header[-1]
    ids_blob = payload[HEADER_STRUCT.size : HEADER_STRUCT.size + ids_length].decode("utf-8")
    floats = payload[HEADER_STRUCT.size + ids_length :]
    ids = ids_blob.split("\0") if ids_blob else []
    return header, ids, floats


def test_encode_agent_frame_metadata():
    positions = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="<f4")
    damage = np.array([0.1, 0.2], dtype="<f4")
    timestamp = 123.456
    frame = encode_agent_frame(
        positions=positions,
        damage_rates=damage,
        agent_ids=["A", "B"],
        timestamp=timestamp,
        components=3,
    )
    header, ids, floats = decode_frame(frame)
    assert header[0] == MAGIC
    assert header[1] == VERSION
    assert header[2] == 2
    assert ids == ["A", "B"]
    float_array = np.frombuffer(floats, dtype="<f4").reshape(-1, 3)
    assert np.allclose(float_array[:, 0:2], positions)
    assert np.allclose(float_array[:, 2], damage)


def test_websocket_stream_broadcasts_ingested_batch():
    client = TestClient(app)
    with client.websocket_connect("/ws/agents") as ws:
        resp = client.post(
            "/ingest",
            json={
                "timestamp": 1.0,
                "agents": [
                    {"agent_id": "agent-1", "x": 0.0, "y": 1.0, "damage": 0.5},
                    {"agent_id": "agent-2", "x": 2.0, "y": 3.0, "damage": 0.2},
                ],
            },
        )
        assert resp.json()["agents"] == 2
        payload = ws.receive_bytes()
        header, ids, floats = decode_frame(payload)
        assert header[2] == 2
        assert ids == ["agent-1", "agent-2"]
        arr = np.frombuffer(floats, dtype="<f4").reshape(-1, 3)
        assert np.isclose(arr[0, 2], 0.5)
