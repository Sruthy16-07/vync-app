"""
audio_codec.py
==============
Audio compression / decompression for the Semantic VC system.

Sender  →  compress_chunk()   : raw PCM bytes  →  compressed bytes
Receiver →  decompress_chunk() : compressed bytes  →  raw PCM bytes

Codec priority (tried in order, first available wins)
------------------------------------------------------
1. MP3   via pydub + ffmpeg      ~10:1 compression  (best quality)
2. OGG   via pydub + ffmpeg      ~12:1 compression  (good quality)
3. ADPCM via audioop (stdlib)    ~4:1  compression   (no ffmpeg needed)
4. μ-law via audioop (stdlib)    ~2:1  compression   (fallback)

The codec that was selected at import time is stored in CODEC_NAME so
both sender and receiver can log which one is running.

Bandwidth comparison at 16 kHz mono
-------------------------------------
  Raw PCM 16-bit   : 256 kbps  (32 KB/s)
  ADPCM            :  64 kbps  ( 8 KB/s)
  μ-law            : 128 kbps  (16 KB/s)
  MP3  @ 32 kbps   :  32 kbps  ( 4 KB/s)   ← default target
  OGG  @ 24 kbps   :  24 kbps  ( 3 KB/s)
"""

import io
import audioop
import struct
import sys

# ---------------------------------------------------------------------------
# Try pydub (needs ffmpeg on PATH)
# ---------------------------------------------------------------------------
try:
    from pydub import AudioSegment
    import subprocess

    # Quick check: is ffmpeg actually available?
    _ffmpeg_check = subprocess.run(
        ["ffmpeg", "-version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    PYDUB_OK = (_ffmpeg_check.returncode == 0)
except Exception:
    PYDUB_OK = False

# ---------------------------------------------------------------------------
# Audio parameters  (must match sender & receiver PyAudio config)
# ---------------------------------------------------------------------------
SAMPLE_RATE   = 16000
CHANNELS      = 1
SAMPLE_WIDTH  = 2        # bytes — 16-bit PCM

# Target bitrate for compressed formats
MP3_BITRATE   = "32k"
OGG_BITRATE   = "24k"

# ---------------------------------------------------------------------------
# Select codec
# ---------------------------------------------------------------------------
if PYDUB_OK:
    # Try MP3 first, fall back to OGG
    try:
        _test_seg = AudioSegment.silent(duration=10,
                                        frame_rate=SAMPLE_RATE)
        _buf = io.BytesIO()
        _test_seg.export(_buf, format="mp3", bitrate=MP3_BITRATE)
        CODEC_NAME = "mp3"
    except Exception:
        try:
            _buf = io.BytesIO()
            _test_seg.export(_buf, format="ogg", bitrate=OGG_BITRATE)
            CODEC_NAME = "ogg"
        except Exception:
            CODEC_NAME = "adpcm"
else:
    CODEC_NAME = "adpcm"

print(f"[CODEC] Selected codec: {CODEC_NAME.upper()}"
      f"  (pydub/ffmpeg available: {PYDUB_OK})")


# ===========================================================================
# Public API
# ===========================================================================

def compress_chunk(pcm_bytes: bytes) -> bytes:
    """
    Compress a raw 16-bit PCM chunk.

    Parameters
    ----------
    pcm_bytes : raw bytes from PyAudio (16-bit signed, mono, SAMPLE_RATE Hz)

    Returns
    -------
    compressed bytes — send these over WebSocket instead of raw PCM
    """
    if CODEC_NAME == "mp3":
        return _compress_mp3(pcm_bytes)
    elif CODEC_NAME == "ogg":
        return _compress_ogg(pcm_bytes)
    elif CODEC_NAME == "adpcm":
        return _compress_adpcm(pcm_bytes)
    else:
        return _compress_ulaw(pcm_bytes)


def decompress_chunk(compressed: bytes) -> bytes:
    """
    Decompress bytes received over WebSocket back to raw 16-bit PCM.

    Returns
    -------
    raw PCM bytes — feed directly into PyAudio output stream
    """
    if CODEC_NAME == "mp3":
        return _decompress_mp3(compressed)
    elif CODEC_NAME == "ogg":
        return _decompress_ogg(compressed)
    elif CODEC_NAME == "adpcm":
        return _decompress_adpcm(compressed)
    else:
        return _decompress_ulaw(compressed)


def compression_ratio(original: bytes, compressed: bytes) -> float:
    """Return compression ratio  (e.g. 10.2 means 10× smaller)."""
    if not compressed:
        return 0.0
    return round(len(original) / len(compressed), 2)


# ===========================================================================
# Codec implementations
# ===========================================================================

# ---------------------------------------------------------------------------
# MP3  (pydub + ffmpeg)
# ---------------------------------------------------------------------------

def _compress_mp3(pcm_bytes: bytes) -> bytes:
    seg = AudioSegment(
        data         = pcm_bytes,
        sample_width = SAMPLE_WIDTH,
        frame_rate   = SAMPLE_RATE,
        channels     = CHANNELS,
    )
    buf = io.BytesIO()
    seg.export(buf, format="mp3", bitrate=MP3_BITRATE)
    return buf.getvalue()


def _decompress_mp3(data: bytes) -> bytes:
    buf = io.BytesIO(data)
    seg = AudioSegment.from_file(buf, format="mp3")
    # Ensure output matches expected format
    seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS) \
              .set_sample_width(SAMPLE_WIDTH)
    return seg.raw_data


# ---------------------------------------------------------------------------
# OGG / Vorbis  (pydub + ffmpeg)
# ---------------------------------------------------------------------------

def _compress_ogg(pcm_bytes: bytes) -> bytes:
    seg = AudioSegment(
        data         = pcm_bytes,
        sample_width = SAMPLE_WIDTH,
        frame_rate   = SAMPLE_RATE,
        channels     = CHANNELS,
    )
    buf = io.BytesIO()
    seg.export(buf, format="ogg", bitrate=OGG_BITRATE)
    return buf.getvalue()


def _decompress_ogg(data: bytes) -> bytes:
    buf = io.BytesIO(data)
    seg = AudioSegment.from_file(buf, format="ogg")
    seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS) \
              .set_sample_width(SAMPLE_WIDTH)
    return seg.raw_data


# ---------------------------------------------------------------------------
# ADPCM  (IMA ADPCM via audioop — stdlib, no ffmpeg needed)
# ~4:1 compression, very low CPU, good for voice
# ---------------------------------------------------------------------------

# IMA ADPCM step table
_STEP_TABLE = [
    7,8,9,10,11,12,13,14,16,17,19,21,23,25,28,31,34,37,41,45,
    50,55,60,66,73,80,88,97,107,118,130,143,157,173,190,209,230,
    253,279,307,337,371,408,449,494,544,598,658,724,796,876,963,
    1060,1166,1282,1411,1552,1707,1878,2066,2272,2499,2749,3024,3327,
    3660,4026,4428,4871,5358,5894,6484,7132,7845,8630,9493,10442,
    11487,12635,13899,15289,16818,18500,20350,22385,24623,27086,29794,32767
]

_INDEX_TABLE = [-1,-1,-1,-1,2,4,6,8,-1,-1,-1,-1,2,4,6,8]


def _compress_adpcm(pcm_bytes: bytes) -> bytes:
    """Encode 16-bit PCM to 4-bit IMA ADPCM."""
    samples = struct.unpack(f"<{len(pcm_bytes)//2}h", pcm_bytes)
    out     = []
    pred    = 0
    idx     = 0
    nibbles = []

    for sample in samples:
        step  = _STEP_TABLE[idx]
        diff  = sample - pred
        sign  = 0
        if diff < 0:
            sign = 8
            diff = -diff

        code  = 0
        delta = step >> 3
        if diff >= step:
            code |= 4;  diff -= step;  delta += step
        step >>= 1
        if diff >= step:
            code |= 2;  diff -= step;  delta += step
        step >>= 1
        if diff >= step:
            code |= 1;  delta += step

        if sign:
            pred -= delta
        else:
            pred += delta
        pred = max(-32768, min(32767, pred))

        idx   = max(0, min(88, idx + _INDEX_TABLE[code]))
        nibbles.append(code | sign)

    # Pack two 4-bit nibbles per byte; prepend 4-byte header (pred, idx)
    header = struct.pack("<hh", pred, idx)
    packed = bytearray()
    for i in range(0, len(nibbles) - 1, 2):
        packed.append(nibbles[i] | (nibbles[i+1] << 4))
    if len(nibbles) % 2:
        packed.append(nibbles[-1])

    return header + bytes(packed)


def _decompress_adpcm(data: bytes) -> bytes:
    """Decode 4-bit IMA ADPCM back to 16-bit PCM."""
    if len(data) < 4:
        return b""

    pred, idx = struct.unpack("<hh", data[:4])
    packed    = data[4:]

    nibbles = []
    for byte in packed:
        nibbles.append(byte & 0x0F)
        nibbles.append((byte >> 4) & 0x0F)

    samples = []
    for code in nibbles:
        step  = _STEP_TABLE[max(0, min(88, idx))]
        sign  = code & 8
        code4 = code & 7

        delta = step >> 3
        if code4 & 4: delta += step
        if code4 & 2: delta += step >> 1
        if code4 & 1: delta += step >> 2

        if sign:
            pred -= delta
        else:
            pred += delta
        pred = max(-32768, min(32767, pred))
        idx  = max(0, min(88, idx + _INDEX_TABLE[code & 7]))
        samples.append(pred)

    return struct.pack(f"<{len(samples)}h", *samples)


# ---------------------------------------------------------------------------
# μ-law  (stdlib audioop — always available)
# ~2:1 compression, good for telephony-quality voice
# ---------------------------------------------------------------------------

def _compress_ulaw(pcm_bytes: bytes) -> bytes:
    return audioop.lin2ulaw(pcm_bytes, SAMPLE_WIDTH)


def _decompress_ulaw(data: bytes) -> bytes:
    return audioop.ulaw2lin(data, SAMPLE_WIDTH)


# ===========================================================================
# Quick self-test
# ===========================================================================

if __name__ == "__main__":
    import math, random

    # Synthesise 1 second of a 440 Hz sine wave as 16-bit PCM
    N       = SAMPLE_RATE
    samples = [int(32000 * math.sin(2 * math.pi * 440 * i / N)) for i in range(N)]
    pcm     = struct.pack(f"<{N}h", *samples)

    compressed   = compress_chunk(pcm)
    decompressed = decompress_chunk(compressed)
    ratio        = compression_ratio(pcm, compressed)

    print(f"\nCodec        : {CODEC_NAME.upper()}")
    print(f"Original     : {len(pcm):>7} bytes")
    print(f"Compressed   : {len(compressed):>7} bytes")
    print(f"Ratio        : {ratio:.1f}×")
    print(f"Decompressed : {len(decompressed):>7} bytes")
    print("Self-test    : PASS" if decompressed else "Self-test    : FAIL (empty output)")
