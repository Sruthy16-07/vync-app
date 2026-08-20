# Semantic Video Conferencing System

Low-bandwidth video conferencing for areas with limited internet connectivity.
Instead of transmitting raw video, the sender analyses the scene and sends a
tiny **Semantic Packet** (~200 bytes / frame).  The receiver reconstructs
animated talking faces from this metadata.

---

## Architecture

```
SENDER                                   RECEIVER
──────                                   ────────
Webcam + Mic                             WebSocket
   │                                         │
   ├─ InsightFace  (face detect + ID)    Parse JSON packet
   ├─ MediaPipe FaceMesh                      │
   │   ├─ geometry.py  (mouth, face dims)  Talking Face Generator
   │   ├─ pose.py      (yaw/pitch/roll)       │  (stub → plug in your model)
   │   └─ expressions.py (emotion)        Display remote participants
   ├─ speaker_detection.py
   ├─ semantic_packet.py  ──── JSON ──►
   └─ WebSocket sender
```

---

## File structure

```
semanxx/
├── main.py               ← SENDER  (run on the caller's machine)
├── receiver.py           ← RECEIVER (run on the callee's machine)
├── speaker_detection.py  ← speaker confidence + VAD
├── semantic_packet.py    ← JSON packet builder / parser
├── face_db.py            ← manual face registration tool
├── face_database.json    ← persisted face embeddings
├── requirements.txt
└── head_pose/
    ├── geometry.py        ← mouth, face width/height, center
    ├── pose.py            ← yaw / pitch / roll estimation
    └── expressions.py     ← lightweight emotion classifier
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# Linux only (for PyAudio):
sudo apt install portaudio19-dev
```

### 2. Register faces (optional, improves recognition)

```bash
python face_db.py
# Press R in front of the camera to register yourself
```

### 3. Start the receiver (on the remote machine)

```bash
python receiver.py --host 0.0.0.0 --port 8765
```

### 4. Start the sender (on your machine)

```bash
# Replace RECEIVER_IP with the receiver's IP address
python main.py --host RECEIVER_IP --port 8765

# Local display only (no network):
python main.py
```

---

## Semantic Packet format

```json
{
    "timestamp": 1719500000.123,
    "people": [
        {
            "id": "P001",
            "emotion": "Happy",
            "emotion_confidence": 0.94,
            "head_pose": { "yaw": 2.1, "pitch": -1.4, "roll": 0.8 },
            "speaker_score": 0.15
        },
        {
            "id": "P002",
            "emotion": "Neutral",
            "emotion_confidence": 0.88,
            "head_pose": { "yaw": 0.3, "pitch": 1.0, "roll": -2.0 },
            "speaker_score": 0.92
        }
    ],
    "active_speaker": "P002"
}
```

---

## Replacing the talking-face stub

`receiver.py` contains a `TalkingFaceGenerator` class with a `render()` method
that currently draws a colour-coded placeholder.  Replace the body of that
method with your preferred model (SadTalker, DiffTalk, FOMM, etc.).  The
method receives all the semantic fields required by any modern talking-face
synthesis approach:

```python
def render(self, person_id, emotion, emotion_confidence,
           yaw, pitch, roll, speaker_score, is_active_speaker) -> np.ndarray:
    # Your model call here – return a BGR (H, W, 3) numpy array
    ...
```

---

## Speaker detection weights

Edit the constants at the top of `speaker_detection.py` to tune detection:

| Weight | Default | Meaning |
|--------|---------|---------|
| W_MOUTH | 0.45 | Mouth movement (strongest cue) |
| W_VOICE | 0.35 | Microphone energy (VAD) |
| W_POSE  | 0.15 | Looking toward camera |
| W_EMOTION | 0.05 | Animated emotion |

---

## `server.py` — Vync frontend bridge

`main.py` assumes it owns the machine's webcam directly via
`cv2.VideoCapture(0)`. That's correct for a desktop app, but a browser tab
can never hand a Python process direct webcam access — only the browser
itself can call `getUserMedia()`. `server.py` is the bridge that makes the
**same pipeline** (InsightFace → MediaPipe FaceMesh → geometry/pose/
expressions → speaker_detection → semantic_packet) work when the camera
lives in a browser tab instead.

```
BROWSER (Vync)                    server.py                    receiver.py
───────────────                   ──────────                  ───────────
getUserMedia()                                                 (run separately,
   │                                                             unchanged)
   ├─ canvas.toBlob() ──JPEG──►  decode (cv2.imdecode)
   │                              │
   │                              ├─ InsightFace (detect + ID)
   │                              ├─ MediaPipe FaceMesh (landmarks)
   │                              ├─ geometry / pose / expressions
   │                              ├─ speaker_detection
   │                              └─ semantic_packet.build_packet()
   │                                          │
   │◄──── echoed back (JSON) ─────────────────┤
   │                                          └──── forwarded ───►  WSReceiver
   └─ Web Audio API ──PCM16──►  VAD energy (compute_voice_energy)
```

### Wire format

The browser and `server.py` share a single WebSocket (`/ws`). Every
**binary** message is prefixed with a 4-byte ASCII tag so the server knows
what it's looking at:

| Tag | Payload | Sent by |
|-----|---------|---------|
| `JPG0` | JPEG-encoded video frame | `vync/src/services/frameCapture.js` |
| `PCM0` | Raw 16-bit PCM mono audio @ 16 kHz | `vync/src/services/frameCapture.js` |

**Text** messages are plain JSON, matching the existing semantic packet
schema documented above, plus chat-style packets (`{"type": "message", ...}`)
sent by `vync/src/services/websocket.js`.

### Running it

```bash
pip install -r requirements.txt
pip install fastapi "uvicorn[standard]"

# Forward semantic packets on to a receiver.py instance running elsewhere:
python server.py --host 0.0.0.0 --port 8000 \
                  --receiver-host RECEIVER_IP --receiver-port 8765

# Or run without forwarding (just exercises the pipeline + echoes packets
# back to the browser, useful for testing the frontend wiring alone):
python server.py --host 0.0.0.0 --port 8000
```

The Vync frontend's default AI server URL
(`DEFAULT_SERVER_URL` in `src/context/MeetingContext.jsx`) is
`ws://192.168.1.15:8000/ws`. Either run `server.py` on a machine reachable
at that address, or change `DEFAULT_SERVER_URL` (or use the in-app device
settings, if `serverUrl` is exposed there) to point at wherever you actually
run it — `ws://localhost:8000/ws` for same-machine development.

### What triggers the pipeline

There is no separate "enable AI" switch. The moment a participant's camera
is live inside an active meeting (`MeetingRoom` mounted, `cameraEnabled`
true, local `MediaStream` available), the frontend's `frameCapture.js`
starts streaming frames/audio to this server automatically, and stops the
instant the camera is turned off or the meeting is left. `server.py` runs
the AI pipeline strictly per-frame as frames arrive — it has no concept of
"meeting state" itself, it just reacts to whatever comes in over the socket.

