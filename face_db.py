"""
face_db.py  –  Manual Face Registration Tool

Run this ONCE per person before starting main.py to pre-register
known faces.  Faces registered here are saved to face_database.json
and loaded automatically by the sender pipeline.

Controls
--------
  r  – register the first detected face under a new ID
  d  – delete the last registered person
  q  – quit
"""

import cv2
import json
import numpy as np
from insightface.app import FaceAnalysis
from numpy.linalg import norm

# =====================================
# CONFIG
# =====================================

DB_FILE              = "face_database.json"
SIMILARITY_THRESHOLD = 0.6

# =====================================
# LOAD DATABASE
# =====================================

try:
    with open(DB_FILE, "r") as f:
        face_db = json.load(f)
    print(f"[DB] Loaded {len(face_db)} existing faces from {DB_FILE}")
except Exception:
    face_db = {}
    print("[DB] No existing database – starting fresh.")

# Convert stored lists back to numpy arrays
for pid in list(face_db.keys()):
    face_db[pid] = np.array(face_db[pid])

# =====================================
# INSIGHTFACE
# =====================================

app = FaceAnalysis()
app.prepare(ctx_id=0)

# =====================================
# HELPERS
# =====================================

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b) + 1e-9)


def find_person(embedding):
    best_score = -1
    best_match = None
    for pid, stored in face_db.items():
        score = cosine_similarity(embedding, stored)
        if score > best_score:
            best_score = score
            best_match = pid
    if best_score >= SIMILARITY_THRESHOLD:
        return best_match, best_score
    return None, best_score


def save_database():
    with open(DB_FILE, "w") as f:
        json.dump({pid: emb.tolist() for pid, emb in face_db.items()}, f)
    print(f"[DB] Saved {len(face_db)} faces to {DB_FILE}")

# =====================================
# WEBCAM
# =====================================

cap = cv2.VideoCapture(0)
print("[INFO] Face registration tool – press R to register, D to delete last, Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        box         = face.bbox.astype(int)
        x1, y1, x2, y2 = box
        embedding   = face.embedding
        pid, score  = find_person(embedding)
        label       = pid if pid else "Unknown"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({score:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"Registered: {list(face_db.keys())}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    cv2.putText(frame, "R: Register  |  D: Delete last  |  Q: Quit",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.imshow("Face Registration", frame)

    key = cv2.waitKey(1) & 0xFF

    # ── Register ──────────────────────────────────────────────────────────────
    if key == ord('r'):
        if not faces:
            print("[WARN] No face detected!")
            continue
        face      = faces[0]
        embedding = face.embedding
        pid, score = find_person(embedding)
        if pid:
            print(f"[INFO] Face already registered as {pid} (score {score:.3f})")
        else:
            new_id           = f"P{len(face_db)+1:03d}"
            face_db[new_id]  = embedding
            save_database()
            print(f"[DB] Registered new face as {new_id}")

    # ── Delete last ───────────────────────────────────────────────────────────
    elif key == ord('d'):
        if face_db:
            last_key = list(face_db.keys())[-1]
            del face_db[last_key]
            save_database()
            print(f"[DB] Deleted {last_key}")
        else:
            print("[WARN] Database is already empty.")

    # ── Quit ──────────────────────────────────────────────────────────────────
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
