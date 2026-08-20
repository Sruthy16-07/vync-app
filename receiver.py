"""
receiver.py  –  RECEIVER

Semantic Video Conferencing System – Receiver Side

Pipeline (matches the project flowchart):
  1. Listen on WebSocket for incoming semantic packets
  2. Parse JSON → extract person IDs, head pose, emotion, active speaker
  3. Feed data into a Talking Face Generator (stub – replace with your
     preferred face-animation model, e.g. SadTalker, DiffTalk, FOMM)
  4. Display animated remote participants in an OpenCV window

Run (on the receiver machine):
    python receiver.py [--host HOST] [--port PORT]

The receiver listens on 0.0.0.0 by default so any sender on the LAN
can connect.  Pass --host 127.0.0.1 to accept local connections only.
"""

import argparse
import asyncio
import json
import queue
import threading
import time

import cv2
import numpy as np

# ── Optional WebSocket ────────────────────────────────────────────────────────
try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    print("[WARNING] 'websockets' not installed.")

# ── Local module ──────────────────────────────────────────────────────────────
from semantic_packet import (
    parse_packet,
    get_active_speaker_data,
)

# =============================================================================
# Talking Face Generator (stub)
# =============================================================================
#
# Replace the body of TalkingFaceGenerator.render() with a call to your
# chosen generative model (SadTalker, DiffTalk, FOMM, etc.).
#
# The stub draws a colour-coded placeholder rectangle so the rest of
# the pipeline can be tested without a GPU model installed.
# =============================================================================

AVATAR_W, AVATAR_H = 256, 256

EMOTION_COLORS = {
    "Happy":     (0,   220,  50),
    "Sad":       (200,  80,   0),
    "Angry":     (0,    0,  240),
    "Surprised": (0,   200, 240),
    "Fearful":   (150,   0, 180),
    "Disgusted": (0,   160, 100),
    "Neutral":   (160, 160, 160),
}


class TalkingFaceGenerator:
    """
    Stub face generator.

    Replace render() with a call to your real face-animation model.
    The method receives all the semantic data needed by any talking-face
    synthesis approach and should return a (H, W, 3) BGR numpy array.
    """

    def render(
        self,
        person_id:          str,
        emotion:            str,
        emotion_confidence: float,
        yaw:                float,
        pitch:              float,
        roll:               float,
        speaker_score:      float,
        is_active_speaker:  bool,
    ) -> np.ndarray:
        """
        Returns a BGR image (AVATAR_H × AVATAR_W × 3) for this person.

        ── Replace this stub with your real model call ──────────────────────
        Example (SadTalker-style, pseudo-code):

            frame = self.model.generate(
                identity_img  = self.id_images[person_id],
                emotion       = emotion,
                head_pose     = (yaw, pitch, roll),
                audio_driven  = is_active_speaker,
            )
            return frame
        ─────────────────────────────────────────────────────────────────────
        """
        canvas = np.zeros((AVATAR_H, AVATAR_W, 3), dtype=np.uint8)
        color  = EMOTION_COLORS.get(emotion, (160, 160, 160))

        # Background tinted by emotion
        canvas[:] = tuple(max(c // 4, 0) for c in color)

        # Face oval
        cx, cy  = AVATAR_W // 2, AVATAR_H // 2
        face_rx = 80
        face_ry = 100

        # Simulate head yaw by shifting the oval slightly
        shift_x = int(yaw * 0.6)
        shift_y = int(pitch * 0.4)
        cv2.ellipse(canvas,
                    (cx + shift_x, cy + shift_y),
                    (face_rx, face_ry), 0, 0, 360, color, -1)

        # Eyes (simple circles)
        eye_y  = cy + shift_y - 25
        eye_lx = cx + shift_x - 28
        eye_rx = cx + shift_x + 28
        cv2.circle(canvas, (eye_lx, eye_y), 10, (30, 30, 30), -1)
        cv2.circle(canvas, (eye_rx, eye_y), 10, (30, 30, 30), -1)

        # Mouth – wider/open when speaking
        mouth_open = int(20 * speaker_score)
        mouth_y    = cy + shift_y + 35
        mouth_w_px = 50
        if mouth_open > 2:
            cv2.ellipse(canvas,
                        (cx + shift_x, mouth_y),
                        (mouth_w_px // 2, mouth_open),
                        0, 0, 180, (30, 30, 30), -1)
        else:
            cv2.line(canvas,
                     (cx + shift_x - mouth_w_px // 2, mouth_y),
                     (cx + shift_x + mouth_w_px // 2, mouth_y),
                     (30, 30, 30), 3)

        # Active-speaker highlight border
        if is_active_speaker:
            cv2.rectangle(canvas, (2, 2),
                          (AVATAR_W - 3, AVATAR_H - 3), (0, 255, 255), 4)

        # Overlay text
        cv2.putText(canvas, person_id,
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(canvas, emotion,
                    (8, AVATAR_H - 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
        cv2.putText(canvas, f"spk:{speaker_score:.2f}",
                    (8, AVATAR_H - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200, 255, 200), 1)

        return canvas


# =============================================================================
# Packet processor
# =============================================================================

def process_packet(packet: dict, generator: TalkingFaceGenerator) -> np.ndarray:
    """
    Parse a semantic packet and produce a composite display frame
    showing all remote participants.
    """
    people         = packet.get("people", [])
    active_speaker = packet.get("active_speaker")

    if not people:
        blank = np.zeros((AVATAR_H, AVATAR_W, 3), dtype=np.uint8)
        cv2.putText(blank, "No participants", (10, AVATAR_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        return blank

    avatars = []
    for person in people:
        pid     = person["id"]
        pose    = person.get("head_pose", {})
        avatar  = generator.render(
            person_id          = pid,
            emotion            = person.get("emotion", "Neutral"),
            emotion_confidence = person.get("emotion_confidence", 0.0),
            yaw                = pose.get("yaw",   0.0),
            pitch              = pose.get("pitch", 0.0),
            roll               = pose.get("roll",  0.0),
            speaker_score      = person.get("speaker_score", 0.0),
            is_active_speaker  = (pid == active_speaker),
        )
        avatars.append(avatar)

    # Tile avatars horizontally
    composite = np.hstack(avatars)

    # Status bar
    ts_str  = time.strftime("%H:%M:%S",
                             time.localtime(packet.get("timestamp", 0)))
    spk_str = f"Active Speaker: {active_speaker or 'none'}"
    bar     = np.zeros((36, composite.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"{ts_str}  |  {spk_str}  |  {len(people)} participant(s)",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)

    return np.vstack([composite, bar])


# =============================================================================
# WebSocket server (runs in asyncio loop in a background thread)
# =============================================================================

class WSReceiver:
    def __init__(self, host: str, port: int):
        self._host   = host
        self._port   = port
        self._queue  = queue.Queue(maxsize=30)
        self._loop   = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True
        )

    def start(self):
        self._thread.start()

    def get_latest(self) -> dict | None:
        """Return the most recent packet, draining stale ones."""
        latest = None
        while not self._queue.empty():
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        async def handler(ws):
            print(f"[WS] Sender connected: {ws.remote_address}")
            try:
                async for raw_msg in ws:
                    try:
                        packet = parse_packet(raw_msg)
                        if not self._queue.full():
                            self._queue.put_nowait(packet)
                    except Exception as exc:
                        print(f"[WS] Parse error: {exc}")
            except Exception as exc:
                print(f"[WS] Connection closed: {exc}")

        async with websockets.serve(handler, self._host, self._port):
            print(f"[WS] Listening on ws://{self._host}:{self._port}")
            await asyncio.Future()   # run forever


# =============================================================================
# Main
# =============================================================================

def run_receiver(args):
    generator = TalkingFaceGenerator()
    last_frame = np.zeros((AVATAR_H + 36, AVATAR_W, 3), dtype=np.uint8)
    cv2.putText(last_frame, "Waiting for sender …",
                (10, AVATAR_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    ws_receiver = None
    if WS_AVAILABLE:
        ws_receiver = WSReceiver(args.host, args.port)
        ws_receiver.start()
    else:
        print("[ERROR] websockets library required. Install with: pip install websockets")

    print("[INFO] Semantic receiver started – press 'q' to quit.")

    while True:
        if ws_receiver:
            packet = ws_receiver.get_latest()
            if packet:
                last_frame = process_packet(packet, generator)

        cv2.imshow("Semantic Receiver – Remote Participants", last_frame)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    print("[INFO] Receiver stopped.")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Semantic Video Conferencing – Receiver"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="WebSocket bind address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8765,
        help="WebSocket port (default: 8765)"
    )
    args = parser.parse_args()
    run_receiver(args)
