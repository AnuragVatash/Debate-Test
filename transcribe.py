# transcribe.py
import time, math
from datetime import timedelta
from faster_whisper import WhisperModel

MODEL  = "deepdml/faster-whisper-large-v3-turbo-ct2"
AUDIO  = "diatrize/audio/debate.wav"
OUTPUT_DIR = "diatrize"

def hms(t):
    """seconds → HH:MM:SS.mmm string"""
    td = timedelta(seconds=t)
    ms = int((t - math.floor(t)) * 1000)  # keep milliseconds
    return f"{str(td)[:-3]}.{ms:03d}"

model = WhisperModel(MODEL, device="cuda", compute_type="int8")

t0 = time.time()
segments, info = model.transcribe(
    AUDIO,
    vad_filter=True,
    word_timestamps=False   # segment-level is plenty for scoring
)
elapsed = time.time() - t0

# Ensure output directory exists and write transcript there for downstream analysis
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_txt = os.path.join(OUTPUT_DIR, "debate_section.txt")
with open(out_txt, "w", encoding="utf-8") as f:
    for seg in segments:
        f.write(f"{hms(seg.start)}\t{hms(seg.end)}\t{seg.text.strip()}\n")

print(
    "Finished •",
    f"RTF ≈ {info.duration / elapsed:.2f}×  ",
    f"({elapsed/60:.1f} min wall-clock for {info.duration/60:.1f} min audio)",
    f"→ saved {out_txt}",
)
