"""
speaker_detection.py

Calculates a speaker confidence score for each detected face.

Inputs per face:
  - mouth_gap        : vertical mouth opening (pixels)
  - mouth_width      : horizontal mouth width  (pixels)
  - head_pose        : (yaw, pitch, roll) from pose.py
  - voice_energy     : float from Voice Activity Detection (0.0 – 1.0)
  - emotion          : optional emotion string from expressions.py

Output:
  - speaker_score    : float 0.0 – 1.0  (higher = more likely speaking)
  - active_speaker   : person_id of the face with the highest score
"""

# ------------------------------------------------------------------
# Weights  (must sum to 1.0)
# ------------------------------------------------------------------

W_MOUTH   = 0.45   # mouth movement is the strongest cue
W_VOICE   = 0.35   # voice energy from VAD
W_POSE    = 0.15   # looking toward camera → more likely the speaker
W_EMOTION = 0.05   # animated emotion can slightly boost score

# ------------------------------------------------------------------
# Emotion boost table
# Emotions that coincide with speech get a small lift
# ------------------------------------------------------------------

EMOTION_BOOST = {
    "Happy":     1.0,
    "Surprised": 0.9,
    "Angry":     0.8,
    "Sad":       0.4,
    "Neutral":   0.5,
    "Fearful":   0.6,
    "Disgusted": 0.5,
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _mouth_score(mouth_gap: float, mouth_width: float) -> float:
    """
    Normalise mouth openness relative to mouth width so the
    score is resolution-independent.

    ratio > 0.25  → clearly open (speaking)
    ratio < 0.05  → closed (silent)
    """
    if mouth_width < 1:
        return 0.0
    ratio = mouth_gap / mouth_width
    # clamp to [0, 1]
    score = min(max(ratio / 0.35, 0.0), 1.0)
    return score


def _pose_score(yaw: float, pitch: float) -> float:
    """
    Faces turned toward the camera are more likely to be the speaker.
    yaw/pitch are in the -100..+100 range used by pose.py.
    """
    yaw_penalty   = min(abs(yaw)   / 40.0, 1.0)
    pitch_penalty = min(abs(pitch) / 40.0, 1.0)
    return 1.0 - (yaw_penalty * 0.6 + pitch_penalty * 0.4)


def _emotion_score(emotion: str) -> float:
    return EMOTION_BOOST.get(emotion, 0.5)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def compute_speaker_score(
    mouth_gap:    float,
    mouth_width:  float,
    head_pose:    tuple,          # (yaw, pitch, roll)
    voice_energy: float,          # 0.0 – 1.0  from VAD
    emotion:      str = "Neutral",
) -> float:
    """
    Returns a float in [0, 1] representing the probability that
    this face belongs to the current speaker.
    """
    yaw, pitch, _roll = head_pose

    m_score = _mouth_score(mouth_gap, mouth_width)
    v_score = min(max(voice_energy, 0.0), 1.0)
    p_score = _pose_score(yaw, pitch)
    e_score = _emotion_score(emotion)

    score = (
        W_MOUTH   * m_score +
        W_VOICE   * v_score +
        W_POSE    * p_score +
        W_EMOTION * e_score
    )

    return round(min(max(score, 0.0), 1.0), 4)


def identify_active_speaker(face_scores: dict) -> str | None:
    """
    Given a dict  { person_id: speaker_score },
    return the person_id with the highest score.
    Returns None if the dict is empty.

    A minimum threshold of 0.25 is required so that a silent
    room does not force-assign an active speaker.
    """
    if not face_scores:
        return None

    best_id    = max(face_scores, key=face_scores.get)
    best_score = face_scores[best_id]

    SILENCE_THRESHOLD = 0.25
    if best_score < SILENCE_THRESHOLD:
        return None

    return best_id


# ------------------------------------------------------------------
# Simple VAD helper  (energy-based, works on raw PCM bytes)
# ------------------------------------------------------------------

def compute_voice_energy(audio_chunk: bytes, sample_width: int = 2) -> float:
    """
    Estimate voice energy from a raw PCM audio chunk.

    Parameters
    ----------
    audio_chunk  : raw PCM bytes (e.g. from PyAudio callback)
    sample_width : bytes per sample (2 = 16-bit PCM, the default)

    Returns
    -------
    energy : float in [0, 1]
    """
    import struct
    import math

    if not audio_chunk:
        return 0.0

    fmt      = f"{len(audio_chunk) // sample_width}h"
    samples  = struct.unpack(fmt, audio_chunk)
    rms      = math.sqrt(sum(s * s for s in samples) / len(samples))

    # 32 767 is max amplitude for 16-bit audio
    MAX_AMP  = 32767.0
    SILENCE  = 300.0     # below this RMS → treat as silence

    if rms < SILENCE:
        return 0.0

    energy = min((rms - SILENCE) / (MAX_AMP - SILENCE), 1.0)
    return round(energy, 4)
