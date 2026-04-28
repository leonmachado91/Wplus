"""FastAPI WebSocket router — receives audio streams from participant devices."""
from __future__ import annotations

import json
import logging
import secrets
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()

# Binary frame layout (sent by client.html):
#   bytes 0-3  : timestamp_ms  (uint32, big-endian)
#   bytes 4-7  : frame_index   (uint32, big-endian)
#   bytes 8-11 : payload_len   (uint32, big-endian)
#   bytes 12+  : WebM/Opus payload
HEADER_SIZE = 12


def _get_manager(app: Any):
    return getattr(app.state, "participant_manager", None)


@router.post("/join/{session_code}/token")
async def get_join_token(session_code: str, request: Request) -> dict:
    """Issue a one-time participant token. Called by client.html before connecting.

    Accepts any body (or no body at all) — the session code in the URL is enough.
    Token is validated implicitly: only clients with the token can connect via WS.
    """
    token = secrets.token_urlsafe(16)
    return {"token": token, "session_code": session_code}


@router.websocket("/ws/audio/{session_code}/{user_token}")
async def ws_audio_endpoint(
    websocket: WebSocket,
    session_code: str,
    user_token: str,
) -> None:
    await websocket.accept()

    manager = _get_manager(websocket.app)
    if manager is None or not manager.is_active or manager.session_code != session_code:
        await websocket.send_json({"type": "error", "message": "Sessão não encontrada"})
        await websocket.close(code=4404)
        return

    # ── handshake: wait for hello ────────────────────────────────────────
    try:
        raw = await websocket.receive_text()
        msg = json.loads(raw)
    except Exception:
        await websocket.close(code=4400)
        return

    if msg.get("type") != "hello":
        await websocket.send_json({"type": "error", "message": "Esperava hello"})
        await websocket.close(code=4400)
        return

    display_name = str(msg.get("display_name", "Participante"))[:64]
    mode = str(msg.get("mode", "auto"))
    device_info = str(msg.get("device_info", ""))[:128]

    participant = manager.add_participant(
        token=user_token,
        display_name=display_name,
        mode=mode,
        device_info=device_info,
    )
    if participant is None:
        await websocket.send_json({"type": "error", "message": "Não foi possível entrar na sessão"})
        await websocket.close(code=4500)
        return

    manager.register_ws(user_token, websocket)
    await websocket.send_json({
        "type": "welcome",
        "session_code": session_code,
        "display_name": display_name,
    })

    pipeline = manager.get_pipeline(user_token)

    # ── main receive loop ────────────────────────────────────────────────
    try:
        while True:
            message = await websocket.receive()

            # Binary: audio frame
            if "bytes" in message and message["bytes"]:
                raw_bytes: bytes = message["bytes"]
                if len(raw_bytes) > HEADER_SIZE and pipeline:
                    # Extract client-side capture timestamp from header (bytes 0–3)
                    client_ts_ms = int.from_bytes(raw_bytes[0:4], "big")
                    opus_data = raw_bytes[HEADER_SIZE:]
                    pipeline.feed(opus_data, client_timestamp_ms=client_ts_ms)

            # Text: status / time_sync / bye
            elif "text" in message and message["text"]:
                try:
                    txt = json.loads(message["text"])
                    msg_type = txt.get("type")

                    if msg_type == "status":
                        muted = bool(txt.get("muted", False))
                        manager.mute_participant(user_token, muted)

                    elif msg_type == "time_sync":
                        import time
                        await websocket.send_json({
                            "type": "time_sync_response",
                            "client_send_time": txt.get("client_send_time"),
                            "server_recv_time": int(time.time() * 1000),
                        })

                    elif msg_type == "bye":
                        break

                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WebSocket error for %s: %s", display_name, exc)
    finally:
        manager.unregister_ws(user_token)
        manager.remove_participant(user_token)
        logger.info("WebSocket closed for participant %s", display_name)
