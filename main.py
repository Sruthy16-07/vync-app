"""
main.py  —  SENDER
==================
Semantic Video Conferencing System

Pipeline
--------
  1.  Open Webcam + Mic
  2.  InsightFace  → detect & identify all faces (P001, P002 …)
  3.  MediaPipe FaceMesh → 468 landmarks per face
  4.  geometry.py  → mouth_gap, mouth_width
  5.  pose.py      → yaw, pitch, roll
  6.  expressions.py → emotion + confidence
  7.  compute_confidence_score() → every person gets a score [0,1]
  8.  Audio thread → room VAD energy
  9.  identify_active_speaker() → max-score person (above VAD gate)
  10. SpeakerAudioRecorder → save audio to recordings/<id>_<ts>.wav
  11. build_packet() → compact JSON semantic packet
  12. WebSocket → send JSON packet  +  raw PCM audio as binary frame
      Receivers hear the active speaker's voice in real time.

Controls:  q = quit

Run
---
  python main.py                          # local display only
  python main.py --host 192.168.1.50      # stream to receiver
  python main.py --host 192.168.1.50 --port 8765
  python main.py --no-audio               # disable mic
"""

import argparse
import asyncio
import json
import math
import os
import sys
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from numpy.linalg import norm

from insightface.app import FaceAnalysis

from head_pose.geometry import (
    face_center, face_height, face_width, mouth_opening, mouth_width,
)
from head_pose.pose import (
    draw_camera_status, draw_pose_text,
    estimate_head_pose, looking_at_camera,
)
from head_pose.expressions import analyze_expression, draw_expression

from speaker_detection import (
    compute_confidence_score,
    compute_voice_energy,
    identify_active_speaker,
    SpeakerAudioRecorder,
    VAD_GATE,
    RECORDINGS_DIR,
)
from semantic_packet import (
    build_packet, build_person_entry, packet_to_json,
)
from audio_codec import compress_chunk, CODEC_NAME

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False
    print("[WARNING] 'websockets' not installed — local-display mode only.")

try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[WARNING] 'pyaudio' not installed — voice energy defaults to 0.")

# ===========================================================================
# Config
# ===========================================================================

DB_FILE              = "face_database.json"
SIMILARITY_THRESHOLD = 0.6

AUDIO_RATE     = 16000
AUDIO_CHUNK    = 1024          # frames per PyAudio buffer
AUDIO_CHANNELS = 1
AUDIO_FORMAT   = 8             # pyaudio.paInt16

# ===========================================================================
# Face database
# ===========================================================================

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

# ===========================================================================
# WebSocket sender
# Sends two frame types per capture cycle:
#   • text frame  — JSON semantic packet
#   • binary frame — raw PCM audio chunk (16-bit mono 16 kHz)
# ===========================================================================

class WSSender:
    def __init__(self, uri: str):
        self._uri    = uri
        self._queue  = asyncio.Queue()
        self._loop   = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)

    def start(self):
        self._thread.start()

    def send_json(self, json_str: str):
        """Enqueue a JSON semantic packet (text frame)."""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, json_str)

    def send_audio(self, compressed: bytes):
        """Enqueue a compressed audio chunk (binary frame)."""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, compressed)

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._loop_body())

    async def _loop_body(self):
        while True:
            try:
                async with websockets.connect(self._uri) as ws:
                    print(f"[WS] Connected to {self._uri}")
                    while True:
                        payload = await self._queue.get()
                        # str → text frame (semantic JSON)
                        # bytes → binary frame (PCM audio)
                        await ws.send(payload)
            except Exception as exc:
                print(f"[WS] {exc}  — retrying in 2 s …")
                await asyncio.sleep(2)

# ===========================================================================
# Audio capture thread
# ===========================================================================

class AudioCapture:
    """
    Captures the microphone in a daemon thread.

    Each PCM chunk is:
      1. Used to compute room voice_energy  (VAD gate)
      2. Forwarded to SpeakerAudioRecorder (local WAV file per speaker)
      3. Forwarded to WSSender as a binary frame so receivers can
         play the active speaker's voice in real time.
    """

    def __init__(self, recorder: SpeakerAudioRecorder,
                 ws_sender: "WSSender | None"):
        self._recorder    = recorder
        self._ws_sender   = ws_sender
        self._energy      = 0.0
        self._active_spk  = None
        self._lock        = threading.Lock()
        self._spk_lock    = threading.Lock()
        self._running     = False
        self._thread      = None

    def start(self):
        if not AUDIO_AVAILABLE:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._recorder.close()

    def set_active_speaker(self, pid):
        with self._spk_lock:
            self._active_spk = pid

    @property
    def voice_energy(self) -> float:
        with self._lock:
            return self._energy

    def _run(self):
        pa     = pyaudio.PyAudio()
        stream = pa.open(
            rate              = AUDIO_RATE,
            channels          = AUDIO_CHANNELS,
            format            = AUDIO_FORMAT,
            input             = True,
            frames_per_buffer = AUDIO_CHUNK,
        )
        while self._running:
            try:
                chunk  = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                energy = compute_voice_energy(chunk, sample_width=2)

                with self._lock:
                    self._energy = energy

                with self._spk_lock:
                    spk = self._active_spk

                # 1. Save locally per speaker (raw PCM for WAV)
                self._recorder.write(chunk, spk)

                # 2. Compress + stream to receivers
                #    Only transmit when someone is actually speaking
                if spk is not None and self._ws_sender and energy > VAD_GATE:
                    compressed = compress_chunk(chunk)
                    self._ws_sender.send_audio(compressed)

            except Exception as exc:
                print(f"[AUDIO] {exc}")

        stream.stop_stream()
        stream.close()
        pa.terminate()

# ===========================================================================
# HUD helpers
# ===========================================================================

def _draw_scoreboard(frame, scores: dict, active_id, voice_energy: float):
    h, w   = frame.shape[:2]
    pW     = 235
    x0, y0 = w - pW - 8, 8
    lh     = 27

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0-6, y0-4),
                  (w-4, y0 + (len(scores)+3)*lh + 8), (18,18,18), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "Speaker Confidence",
                (x0, y0+lh-6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200,200,200), 1)

    # VAD bar
    vc     = (0,210,0) if voice_energy >= VAD_GATE else (60,60,200)
    bmax   = pW - 20
    blen   = int(bmax * min(voice_energy, 1.0))
    by     = y0 + lh + 4
    cv2.rectangle(frame, (x0, by), (x0+bmax, by+10), (50,50,50), -1)
    cv2.rectangle(frame, (x0, by), (x0+blen, by+10), vc, -1)
    cv2.putText(frame, f"VAD {voice_energy:.2f}  {'ACTIVE' if voice_energy>=VAD_GATE else 'SILENT'}",
                (x0, by+22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,180), 1)

    for i, (pid, sc) in enumerate(sorted(scores.items(), key=lambda x:-x[1])):
        ry     = y0 + (i+3)*lh
        is_spk = (pid == active_id)
        col    = (0,255,255) if is_spk else (150,150,150)
        prefix = "SPEAK " if is_spk else "      "

        blen2 = int((pW-85) * sc)
        cv2.rectangle(frame,(x0+78,ry-11),(x0+78+pW-85,ry-3),(40,40,40),-1)
        cv2.rectangle(frame,(x0+78,ry-11),(x0+78+blen2,ry-3),col,-1)
        cv2.putText(frame, f"{prefix}{pid} {sc:.2f}",
                    (x0, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col,
                    2 if is_spk else 1)

# ===========================================================================
# Main sender pipeline
# ===========================================================================

def run_sender(args):
    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    face_db    = load_face_db()
    recorder   = SpeakerAudioRecorder(rate=AUDIO_RATE, channels=AUDIO_CHANNELS)

    # WebSocket sender (created before AudioCapture so the ref can be passed)
    ws_sender = None
    if WS_AVAILABLE and args.host:
        ws_sender = WSSender(f"ws://{args.host}:{args.port}")
        ws_sender.start()

    audio = AudioCapture(recorder, ws_sender)
    if not args.no_audio:
        audio.start()

    # InsightFace
    insight_app = FaceAnalysis()
    insight_app.prepare(ctx_id=0)

    # MediaPipe FaceMesh
    mp_fm    = mp.solutions.face_mesh
    face_mesh = mp_fm.FaceMesh(
        static_image_mode        = False,
        max_num_faces            = 6,
        refine_landmarks         = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence  = 0.5,
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("[ERROR] Cannot open webcam.")

    print("[INFO] Sender running — press 'q' to quit.")
    print(f"[INFO] Recordings → ./{RECORDINGS_DIR}/")
    if ws_sender:
        print(f"[INFO] Streaming to ws://{args.host}:{args.port}")
        print(f"[INFO] Audio codec : {CODEC_NAME.upper()} (compressed binary frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]

        # ── InsightFace: detect + identify ───────────────────────────────────
        insight_faces  = insight_app.get(frame)
        insight_labels = []   # [(cx, cy, person_id, bbox)]

        for face in insight_faces:
            pid, sim = find_person(face.embedding, face_db)
            if pid is None:
                pid           = f"P{len(face_db)+1:03d}"
                face_db[pid]  = face.embedding
                save_face_db(face_db)
                print(f"[DB] New person: {pid}")

            box             = face.bbox.astype(int)
            x1,y1,x2,y2    = box
            cx,cy           = (x1+x2)//2, (y1+y2)//2
            insight_labels.append((cx,cy,pid,box))

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,220,0),2)
            cv2.putText(frame,pid,(x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,220,0),2)

        # ── FaceMesh: landmarks → scores ─────────────────────────────────────
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        voice_energy   = audio.voice_energy
        confidence_map = {}
        people_entries = []

        if results.multi_face_landmarks:
            for idx, lms in enumerate(results.multi_face_landmarks):

                # Correlate mesh → InsightFace identity
                mx, my     = face_center(lms, w, h)
                matched_id = None
                best_dist  = float("inf")
                for (cx,cy,pid,_) in insight_labels:
                    d = math.hypot(mx-cx, my-cy)
                    if d < best_dist:
                        best_dist, matched_id = d, pid
                if matched_id is None:
                    matched_id = f"Unknown_{idx}"

                # Geometry
                m_gap  = mouth_opening(lms, w, h)
                m_wid  = mouth_width(lms, w, h)
                center = face_center(lms, w, h)

                # Head pose
                yaw, pitch, roll = estimate_head_pose(lms, w, h)
                looking          = looking_at_camera(yaw, pitch)

                # Emotion
                expr     = analyze_expression(lms, w, h)
                emotion  = expr["emotion"]
                emo_conf = expr["confidence"]

                # Confidence score for this face
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

                mp_draw.draw_landmarks(frame, lms, mp_fm.FACEMESH_TESSELATION)

                y_off = idx * 160
                cv2.putText(frame,f"ID   : {matched_id}",(14,32+y_off),
                            cv2.FONT_HERSHEY_SIMPLEX,0.58,(255,140,0),2)
                cv2.putText(frame,f"Conf : {conf:.3f}",(14,57+y_off),
                            cv2.FONT_HERSHEY_SIMPLEX,0.58,(0,255,255),2)
                cv2.putText(frame,f"Mouth: gap {m_gap:.1f}  w {m_wid:.1f}",(14,78+y_off),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,210,0),1)
                cv2.putText(frame,f"Pose : y{yaw:.1f} p{pitch:.1f} r{roll:.1f}",(14,96+y_off),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,210,0),1)
                cv2.putText(frame,f"Emo  : {emotion} ({emo_conf:.2f})",(14,114+y_off),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,0),1)
                cv2.circle(frame, center, 4, (0,0,255), -1)

        # ── Active speaker selection ──────────────────────────────────────────
        active_speaker, confidence_map = identify_active_speaker(
            confidence_map, voice_energy
        )

        # ── Tell audio thread who is speaking ─────────────────────────────────
        # AudioCapture._run() will route the next PCM chunk to:
        #   a) the local WAV file for that speaker ID
        #   b) the WebSocket as a binary frame → receivers play it
        audio.set_active_speaker(active_speaker)

        # ── HUD ──────────────────────────────────────────────────────────────
        _draw_scoreboard(frame, confidence_map, active_speaker, voice_energy)

        if active_speaker:
            cv2.putText(frame, f"SPEAKING: {active_speaker}",
                        (w//2-130, h-18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,255), 2)
            for (cx,cy,pid,box) in insight_labels:
                if pid == active_speaker:
                    x1,y1,x2,y2 = box
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,0),3)

        # ── Semantic packet → WebSocket text frame ────────────────────────────
        packet      = build_packet(people_entries, active_speaker)
        packet_json = packet_to_json(packet)
        if ws_sender:
            ws_sender.send_json(packet_json)

        cv2.imshow("Semantic Sender", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ───────────────────────────────────────────────────────────────
    audio.stop()
    cap.release()
    cv2.destroyAllWindows()

    log = recorder.session_log()
    if log:
        print("\n[SESSION] Recorded audio segments:")
        for pid, path, dur in log:
            print(f"  {pid:>6}  {dur:>6.1f}s  →  {path}")
    else:
        print("[SESSION] No audio segments recorded.")
    print("[INFO] Sender stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic VC — Sender")
    parser.add_argument("--host", type=str, default=None,
        help="Receiver IP (omit for local-only mode)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-audio", action="store_true")
    run_sender(parser.parse_args())
