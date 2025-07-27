#!/usr/bin/env python3
import os, sys, json, tempfile, subprocess
from pyannote.audio import Pipeline
import whisper

def diarize(audio_path, token):
    """
    Run speaker diarization to obtain a list of (start, end, speaker) tuples.
    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token
    )                                          # load diarization pipeline :contentReference[oaicite:6]{index=6}
    diarization = pipeline(audio_path)         # process audio.wav :contentReference[oaicite:7]{index=7}
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end":   turn.end,
            "speaker": speaker
        })
    return segments

def transcribe_segments(audio_path, segments):
    """
    For each diarization segment, extract audio and run Whisper transcription.
    """
    model = whisper.load_model("medium.en")    # load medium English model :contentReference[oaicite:8]{index=8}
    results = []
    for seg in segments:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        # extract segment using ffmpeg-python
        (
            subprocess.run([
                "ffmpeg", "-y",
                "-i", audio_path,
                "-ss", str(seg["start"]),
                "-to", str(seg["end"]),
                "-ar", "16000",
                "-ac", "1",
                tmp_path
            ], capture_output=True)
        )                                         # uses FFmpeg directly to trim :contentReference[oaicite:9]{index=9}
        # transcribe with Whisper
        res = model.transcribe(tmp_path)
        results.append({
            "speaker": seg["speaker"],
            "start":   seg["start"],
            "end":     seg["end"],
            "text":    res["text"].strip()
        })
        os.remove(tmp_path)
    return results

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 diarize_and_transcribe.py <audio.wav>")
        sys.exit(1)

    audio_file = sys.argv[1]
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("Error: set HUGGINGFACE_TOKEN environment variable")
        sys.exit(1)

    # 1. Diarize
    segments = diarize(audio_file, hf_token)

    # 2. Transcribe each segment
    annotated = transcribe_segments(audio_file, segments)

    # 3. Output JSON
    output_path = os.path.splitext(audio_file)[0] + "_diarized.json"
    with open(output_path, "w") as f:
        json.dump(annotated, f, indent=2)
    print(f"Saved diarized transcript to {output_path}")

if __name__ == "__main__":
    main()
