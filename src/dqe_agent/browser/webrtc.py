"""WebRTC streaming — CDP screencast JPEG frames → WebRTC video track → browser.

Flow:
  Chromium CDP → JPEG bytes → BrowserVideoTrack → RTCPeerConnection → <video>

Signaling over the existing WebSocket:
  Client → {"type": "webrtc_offer", "sdp": "..."}
  Server → {"type": "webrtc_answer", "sdp": "..."}
"""
from __future__ import annotations

import asyncio
import base64
import fractions
import io
import logging
import time
from typing import Any

try:
    import av
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import MediaStreamTrack
    from av import VideoFrame
    _WEBRTC_OK = True
except ImportError as _webrtc_import_err:
    _WEBRTC_OK = False
    RTCPeerConnection = None  # type: ignore[assignment,misc]
    RTCSessionDescription = None  # type: ignore[assignment,misc]
    MediaStreamTrack = object  # type: ignore[assignment,misc]
    VideoFrame = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)
if not _WEBRTC_OK:
    logger.warning("[WebRTC] av/aiortc not available — WebRTC disabled. Install with: pip install av aiortc")

# Active peer connections: session_id → RTCPeerConnection
_peer_connections: dict[str, RTCPeerConnection] = {}


class BrowserVideoTrack(MediaStreamTrack):
    """Video track fed by CDP screencast JPEG frames at ~30 fps."""

    kind = "video"

    def __init__(self) -> None:
        super().__init__()
        self._latest_frame: VideoFrame | None = None
        self._frame_event = asyncio.Event()
        self._start = time.time()

    def push_frame(self, jpeg_bytes: bytes) -> None:
        """Decode JPEG and store as latest frame. Called from CDP callback."""
        try:
            container = av.open(io.BytesIO(jpeg_bytes), format="mjpeg")
            for frame in container.decode(video=0):
                self._latest_frame = frame
                self._frame_event.set()
                break
            container.close()
        except Exception as e:
            logger.debug("[WebRTC] frame decode error: %s", e)

    async def recv(self) -> VideoFrame:
        """Called by aiortc to get the next frame at ~30 fps.

        CDP screencast only sends frames on page changes, so on static pages we
        re-send the last frame to keep the stream alive. The sleep goes FIRST so
        aiortc never waits more than 33 ms regardless of page activity.
        """
        await asyncio.sleep(1 / 30)
        if self._latest_frame is None:
            # No frame yet — wait (only on first call before any page renders)
            await self._frame_event.wait()
        frame = self._latest_frame
        # Build a new VideoFrame with the right PTS so we don't mutate the
        # shared frame object and confuse the encoder on concurrent recv calls.
        out = frame.reformat(format=frame.format.name)
        out.pts = int((time.time() - self._start) * 30)
        out.time_base = fractions.Fraction(1, 30)
        return out


async def handle_webrtc_offer(session_id: str, sdp: str, session_bm: Any) -> dict:
    """Handle a WebRTC offer from the frontend and return an answer."""
    if not _WEBRTC_OK:
        return {"type": "error", "message": "WebRTC unavailable: av/aiortc not installed"}
    try:
        await cleanup_webrtc(session_id)

        pc = RTCPeerConnection()
        _peer_connections[session_id] = pc

        track = BrowserVideoTrack()

        async def _feed(b64: str) -> None:
            track.push_frame(base64.b64decode(b64))

        session_bm.add_frame_callback(_feed)
        pc._dqe_cb = _feed          # type: ignore[attr-defined]
        pc._dqe_bm = session_bm     # type: ignore[attr-defined]

        pc.addTrack(track)

        @pc.on("connectionstatechange")
        async def _on_state() -> None:
            logger.info("[WebRTC/%s] state → %s", session_id, pc.connectionState)
            if pc.connectionState in ("failed", "closed", "disconnected"):
                await cleanup_webrtc(session_id)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        logger.info("[WebRTC/%s] connection established", session_id)
        return {"type": "webrtc_answer", "sdp": pc.localDescription.sdp}

    except Exception as exc:
        logger.error("[WebRTC/%s] offer failed: %s", session_id, exc)
        return {"type": "error", "message": f"WebRTC setup failed: {exc}"}


async def cleanup_webrtc(session_id: str) -> None:
    """Close and remove the peer connection for a session."""
    pc = _peer_connections.pop(session_id, None)
    if pc is None:
        return
    cb = getattr(pc, "_dqe_cb", None)
    bm = getattr(pc, "_dqe_bm", None)
    if cb and bm:
        bm.remove_frame_callback(cb)
    try:
        await pc.close()
    except Exception:
        pass
    logger.info("[WebRTC/%s] cleaned up", session_id)
