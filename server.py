"""
server.py  —  BROWSER-FACING BRIDGE
=====================================
Semantic Video Conferencing System — FastAPI WebSocket bridge

This is the piece that connects the Vync React frontend to the existing
semanxx AI pipeline (InsightFace + MediaPipe + speaker_detection). It does
NOT replace main.py's logic — it reuses it. The only thing that changes is
where the video frames come from:

    main.py    : frames come from cv2.VideoCapture(0)   (local webcam)
    server.py  : frames come from the browser, one JPEG per WebSocket
                 binary message, sent by the React app's frame-capture
                 service while the meeting is live.

Per browser connection, this module:
  1. Accepts a WebSocket at /ws  (matches src/services/websocket.js's
     DEFAULT_SERVER_URL = "ws://<host>:8000/ws")
  2. Receives:
       - binary frames  → JPEG-encoded video frame  → run AI pipeline
       - binary frames prefixed with the 4-byte tag b"PCM0" → raw 16-bit
         PCM audio chunk (16 kHz mono) → feed into speaker_detection VAD
       - text frames     → JSON control/chat packets, shape
         { type: "message" | ... }  (passed straight to the
         message-handling side, unrelated to the AI pipeline)
  3. Runs the SAME pipeline steps as main.py, per incoming video frame:
       InsightFace (detect + identify) → MediaPipe FaceMesh (landmarks)
       → geometry.py (mouth) → pose.py (yaw/pitch/roll)
       → expressions.py (emotion) → speaker_detection.compute_confidence_score
       → speaker_detection.identify_active_speaker
       → semantic_packet.build_packet
  4. Sends the resulting semantic packet back down the SAME browser
     WebSocket (so the Vync UI can show connection/status), and ALSO
     forwards it on to the separately-handled receiver server using the
     existing WSSender from main.py, completely unchanged.

Run
---
    pip install -r requirements.txt
    pip install fastapi "uvicorn[standard]"
    python server.py --host 0.0.0.0 --port 8000 --receiver-host RECEIVER_IP --receiver-port 8765

The frontend's default serverUrl (see MeetingContext.jsx) is
ws://192.168.1.15:8000/ws — point --host/--port so that the machine
running this server is reachable at that address, or change
DEFAULT_SERVER_URL in the frontend to match wherever this runs.
"""

import argparse
import asyncio
import io
import json
import math
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from numpy.linalg import norm

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from insightface.app import FaceAnalysis

from head_pose.geometry import face_center, mouth_opening, mouth_width
from head_pose.pose import estimate_head_pose
from head_pose.expressions import analyze_expression

from speaker_detection import (
    compute_confidence_score,
    identify_active_speaker,
    compute_voice_energy,
    VAD_GATE,
)
from semantic_packet import build_packet, build_person_entry, packet_to_json

# ===========================================================================
# Binary frame tags
# ---------------------------------------------------------------------------
# The frontend sends two different kinds of binary WebSocket messages over
# the SAME socket (video frames and audio chunks). A 4-byte ASCII tag at
# the start of each binary message tells us which one we're looking at.
# Keep these in sync with src/services/frameCapture.js on the frontend.
# ===========================================================================
TAG_VIDEO = b"JPG0"
TAG_AUDIO = b"PCM0"
TAG_LEN   = 4

# ===========================================================================
# Face database (shared across all connections, same file main.py /
# face_db.py use, loaded once at startup)
# ===========================================================================
DB_FILE              = "face_database.json"
SIMILARITY_THRESHOLD = 0.6


def load_face_db() -> dict:
    try:
        with open(DB_FILE) as f:
            raw = json.load(f)
        db = {pid: np.array(emb) for pid, emb in raw.items()}
        print(f"[DB] Loaded {len(db)} registered faces.")
        return db
    except Exception:
        print("[DB] No database found — starting fresh.")
        return {}


def save_face_db(db: dict):
    with open(DB_FILE, "w") as f:
        json.dump({pid: emb.tolist() for pid, emb in db.items()}, f)


def cosine_sim(a, b) -> float:
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-9))


def find_person(embedding, db: dict):
    best_score, best_id = -1.0, None
    for pid, stored in db.items():
        s = cosine_sim(embedding, stored)
        if s > best_score:
            best_score, best_id = s, pid
    if best_score >= SIMILARITY_THRESHOLD:
        return best_id, best_score
    return None, best_score


FACE_DB = load_face_db()

# ===========================================================================
# Shared model instances
# ---------------------------------------------------------------------------
# InsightFace and MediaPipe are expensive to construct (loads ONNX models
# etc.), so — exactly like main.py — they are created once and reused, not
# per-connection and not per-frame.
# ===========================================================================
print("[INIT] Loading InsightFace …")
INSIGHT_APP = FaceAnalysis()
INSIGHT_APP.prepare(ctx_id=0)

print("[INIT] Loading MediaPipe FaceMesh …")
_mp_fm = mp.solutions.face_mesh
FACE_MESH = _mp_fm.FaceMesh(
    static_image_mode        = False,
    max_num_faces             = 6,
    refine_landmarks          = True,
    min_detection_confidence  = 0.5,
    min_tracking_confidence   = 0.5,
)

# ===========================================================================
# Forwarding to the separately-handled receiver
# ---------------------------------------------------------------------------
# This is main.py's WSSender, unchanged. The bridge server uses it to push
# every semantic packet it builds onward to whatever receiver process is
# listening (the one started with receiver.py, run independently).
# ===========================================================================

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    print("[WARNING] 'websockets' not installed — cannot forward to receiver.")


class WSSender:
    """Unchanged from main.py — forwards JSON text frames to the receiver."""

    def __init__(self, uri: str):
        self._uri    = uri
        self._queue  = asyncio.Queue()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._task: Optional[asyncio.Task] = None

    def attach_to_running_loop(self, loop: asyncio.AbstractEventLoop):
        """Bind to the asyncio loop FastAPI/uvicorn is already running."""
        self._loop = loop
        self._task = loop.create_task(self._loop_body())

    def send_json(self, json_str: str):
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, json_str)

    def send_audio(self, compressed: bytes):
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._queue.put_nowait, compressed)

    async def _loop_body(self):
        while True:
            try:
                async with websockets.connect(self._uri) as ws:
                    print(f"[WS→RECEIVER] Connected to {self._uri}")
                    while True:
                        payload = await self._queue.get()
                        await ws.send(payload)
            except Exception as exc:
                print(f"[WS→RECEIVER] {exc} — retrying in 2 s …")
                await asyncio.sleep(2)


RECEIVER_SENDER: Optional[WSSender] = None  # created in main() once args are known

# ===========================================================================
# Per-connection AI pipeline state
# ---------------------------------------------------------------------------
# Each browser tab that joins a meeting opens its own WebSocket connection,
# and gets its own VoiceState (room VAD energy) since each browser captures
# its own microphone. InsightFace/MediaPipe/FACE_DB are shared (see above)
# because identity recognition should work across all connections.
# ===========================================================================

class ConnectionState:
    def __init__(self):
        self.voice_energy = 0.0


def process_video_frame(frame_bytes: bytes, state: ConnectionState) -> dict:
    """
    Run the exact same pipeline main.py runs per webcam frame, but on a
    single JPEG frame decoded from the browser instead of a cv2.VideoCapture
    read. Returns a semantic packet dict (see semantic_packet.py).
    """
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return build_packet([], None)

    h, w = frame.shape[:2]

    # ── InsightFace: detect + identify ────────────────────────────────────
    insight_faces  = INSIGHT_APP.get(frame)
    insight_labels = []  # [(cx, cy, person_id)]

    for face in insight_faces:
        pid, _sim = find_person(face.embedding, FACE_DB)
        if pid is None:
            pid = f"P{len(FACE_DB) + 1:03d}"
            FACE_DB[pid] = face.embedding
            save_face_db(FACE_DB)
            print(f"[DB] New person: {pid}")

        box = face.bbox.astype(int)
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        insight_labels.append((cx, cy, pid))

    # ── FaceMesh: landmarks → scores ───────────────────────────────────────
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = FACE_MESH.process(rgb)

    confidence_map = {}
    people_entries = []

    if results.multi_face_landmarks:
        for idx, lms in enumerate(results.multi_face_landmarks):

            # Correlate mesh → InsightFace identity (nearest center)
            mx, my     = face_center(lms, w, h)
            matched_id = None
            best_dist  = float("inf")
            for (cx, cy, pid) in insight_labels:
                d = math.hypot(mx - cx, my - cy)
                if d < best_dist:
                    best_dist, matched_id = d, pid
            if matched_id is None:
                matched_id = f"Unknown_{idx}"

            # Geometry
            m_gap  = mouth_opening(lms, w, h)
            m_wid  = mouth_width(lms, w, h)

            # Head pose
            yaw, pitch, roll = estimate_head_pose(lms, w, h)

            # Emotion
            expr     = analyze_expression(lms, w, h)
            emotion  = expr["emotion"]
            emo_conf = expr["confidence"]

            # Speaker confidence for this face
            conf = compute_confidence_score(
                mouth_gap   = m_gap,
                mouth_width = m_wid,
                head_pose   = (yaw, pitch, roll),
                emotion     = emotion,
            )
            confidence_map[matched_id] = conf

            people_entries.append(build_person_entry(
                person_id=matched_id, emotion=emotion,
                emotion_confidence=emo_conf,
                yaw=yaw, pitch=pitch, roll=roll,
                speaker_score=conf,
            ))

    # ── Active speaker selection (same VAD-gated logic as main.py) ─────────
    active_speaker, confidence_map = identify_active_speaker(
        confidence_map, state.voice_energy
    )

    return build_packet(people_entries, active_speaker)


# ===========================================================================
# FastAPI app
# ===========================================================================

app = FastAPI(title="Semanxx Bridge Server")

# Permissive CORS: the React dev server and the bridge run on different
# ports/hosts. WebSocket upgrade requests aren't restricted by CORS in the
# same way XHR is, but the browser still checks the Origin on the initial
# HTTP handshake, so this keeps local dev friction-free. Lock this down to
# your actual frontend origin before deploying anywhere reachable publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "registered_faces": len(FACE_DB)}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"[WS] Browser connected: {websocket.client}")

    state = ConnectionState()

    try:
        while True:
            message = await websocket.receive()

            # FastAPI/Starlette WebSocket.receive() returns a dict with
            # either "text" or "bytes" depending on the frame type the
            # browser sent (matches plain WebSocket.send / send(blob)).
            if "bytes" in message and message["bytes"] is not None:
                raw = message["bytes"]
                tag, payload = raw[:TAG_LEN], raw[TAG_LEN:]

                if tag == TAG_VIDEO:
                    packet = process_video_frame(payload, state)
                    packet_json = packet_to_json(packet)

                    # 1) Echo back to this same browser tab, so the Vync UI
                    #    can reflect connection/activity status if it wants to.
                    await websocket.send_text(packet_json)

                    # 2) Forward onward to the separately-handled receiver,
                    #    exactly as main.py's sender loop does.
                    if RECEIVER_SENDER is not None:
                        RECEIVER_SENDER.send_json(packet_json)

                elif tag == TAG_AUDIO:
                    # Raw 16-bit PCM mono chunk from the browser mic.
                    state.voice_energy = compute_voice_energy(payload)

                    if RECEIVER_SENDER is not None:
                        RECEIVER_SENDER.send_audio(payload)

                else:
                    print(f"[WS] Unknown binary tag: {tag!r} — ignoring frame.")

            elif "text" in message and message["text"] is not None:
                # Non-AI control/chat packets (type: "message", etc.) — not
                # part of the vision pipeline, just acknowledged here so the
                # existing chat flow in websocket.js keeps working end to end
                # if this same endpoint is reused for chat relay later.
                try:
                    json.loads(message["text"])
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        print(f"[WS] Browser disconnected: {websocket.client}")
    except Exception as exc:
        print(f"[WS] Connection error: {exc}")


@app.on_event("startup")
async def on_startup():
    global RECEIVER_SENDER
    if RECEIVER_SENDER is not None:
        RECEIVER_SENDER.attach_to_running_loop(asyncio.get_running_loop())


def main():
    parser = argparse.ArgumentParser(description="Semanxx FastAPI bridge server")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                         help="Bind address for the browser-facing server")
    parser.add_argument("--port", type=int, default=8000,
                         help="Bind port for the browser-facing server "
                              "(must match serverUrl in the frontend, default 8000)")
    parser.add_argument("--receiver-host", type=str, default=None,
                         help="IP of the separately-run receiver.py process. "
                              "Omit to disable forwarding (bridge will still "
                              "echo semantic packets back to the browser).")
    parser.add_argument("--receiver-port", type=int, default=8765,
                         help="Port the receiver.py process is listening on")
    args = parser.parse_args()

    global RECEIVER_SENDER
    if WS_AVAILABLE and args.receiver_host:
        RECEIVER_SENDER = WSSender(f"ws://{args.receiver_host}:{args.receiver_port}")
        print(f"[INFO] Will forward semantic packets to "
              f"ws://{args.receiver_host}:{args.receiver_port}")
    else:
        print("[INFO] No --receiver-host given — packets will only be echoed "
              "back to the browser, not forwarded to a receiver.")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
