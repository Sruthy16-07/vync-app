"""
semantic_packet.py

Builds and parses the Semantic Packet – the compact JSON payload
that is transmitted over WebSocket instead of a full video stream.

Packet schema
-------------
{
    "timestamp": <float>,          # Unix time (seconds)
    "people": [
        {
            "id":                  <str>,    # e.g. "P001"
            "emotion":             <str>,    # e.g. "Happy"
            "emotion_confidence":  <float>,  # 0.0 – 1.0
            "head_pose": {
                "yaw":   <float>,
                "pitch": <float>,
                "roll":  <float>
            },
            "speaker_score":       <float>   # 0.0 – 1.0
        },
        ...
    ],
    "active_speaker": <str | null>   # person_id or null
}
"""

import json
import time


# ------------------------------------------------------------------
# Build
# ------------------------------------------------------------------

def build_person_entry(
    person_id:          str,
    emotion:            str,
    emotion_confidence: float,
    yaw:                float,
    pitch:              float,
    roll:               float,
    speaker_score:      float,
) -> dict:
    """Construct a single person block for the semantic packet."""
    return {
        "id":                 person_id,
        "emotion":            emotion,
        "emotion_confidence": round(emotion_confidence, 4),
        "head_pose": {
            "yaw":   round(yaw,   2),
            "pitch": round(pitch, 2),
            "roll":  round(roll,  2),
        },
        "speaker_score": round(speaker_score, 4),
    }


def build_packet(people: list[dict], active_speaker: str | None) -> dict:
    """
    Assemble the complete semantic packet.

    Parameters
    ----------
    people          : list of dicts produced by build_person_entry()
    active_speaker  : person_id string or None

    Returns
    -------
    packet : dict  (ready to JSON-serialise)
    """
    return {
        "timestamp":      time.time(),
        "people":         people,
        "active_speaker": active_speaker,
    }


def packet_to_json(packet: dict) -> str:
    """Serialise packet to a compact JSON string (no indent)."""
    return json.dumps(packet, separators=(",", ":"))


def packet_to_json_pretty(packet: dict) -> str:
    """Serialise packet to a human-readable JSON string."""
    return json.dumps(packet, indent=4)


# ------------------------------------------------------------------
# Parse (receiver side)
# ------------------------------------------------------------------

def parse_packet(raw: str | bytes) -> dict:
    """
    Deserialise a JSON semantic packet received over WebSocket.

    Returns
    -------
    packet : dict  with keys  timestamp, people, active_speaker
    """
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def get_person(packet: dict, person_id: str) -> dict | None:
    """Return the person block for a given ID, or None."""
    for p in packet.get("people", []):
        if p["id"] == person_id:
            return p
    return None


def get_active_speaker_data(packet: dict) -> dict | None:
    """Return the full person block for the active speaker, or None."""
    active_id = packet.get("active_speaker")
    if active_id is None:
        return None
    return get_person(packet, active_id)


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    p1 = build_person_entry(
        person_id="P001",
        emotion="Happy",
        emotion_confidence=0.94,
        yaw=2.1, pitch=-1.4, roll=0.8,
        speaker_score=0.15,
    )
    p2 = build_person_entry(
        person_id="P002",
        emotion="Neutral",
        emotion_confidence=0.88,
        yaw=0.3, pitch=1.0, roll=-2.0,
        speaker_score=0.92,
    )
    packet = build_packet(people=[p1, p2], active_speaker="P002")
    print(packet_to_json_pretty(packet))
