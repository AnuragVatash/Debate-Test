#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent
DIATRIZE = ROOT / "diatrize"
AUDIO_DIR = DIATRIZE / "audio"
DEBATE_DIR = DIATRIZE / "debate"
SCORED_DIR = DIATRIZE / "scored"
JUSTIFY_DIR = DIATRIZE / "justify"


def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9\-_.]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "debate"


def ensure_dirs() -> None:
    for d in (AUDIO_DIR, DEBATE_DIR, SCORED_DIR, JUSTIFY_DIR):
        d.mkdir(parents=True, exist_ok=True)


def to_hms(total_seconds: int) -> str:
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_duration(s: str) -> Optional[str]:
    s = s.strip()
    if not s:
        return None
    if re.match(r"^\d{1,2}:\d{2}:\d{2}$", s):
        return s
    # seconds
    if s.isdigit():
        return to_hms(int(s))
    return None


def run(cmd: list[str], input_text: Optional[str] = None, cwd: Optional[Path] = None) -> None:
    print("$", " ".join(cmd))
    res = subprocess.run(
        cmd,
        input=(input_text.encode("utf-8") if input_text is not None else None),
        cwd=str(cwd) if cwd else None,
        stdout=sys.stdout,
        stderr=sys.stderr,
        check=True,
    )


def download_audio(url: str, out_wav: Path, end_hms: Optional[str]) -> None:
    # Requires: yt-dlp, ffmpeg
    sections = ["--download-sections", f"*0-{end_hms}"] if end_hms else []
    run([
        "yt-dlp",
        *sections,
        "-x", "--audio-format", "wav",
        "--postprocessor-args", "ffmpeg:-ac 1 -ar 16000",
        "-o", str(out_wav.with_suffix(".%(ext)s")),
        url,
    ])
    # If yt-dlp didn't produce exact .wav at target name, try to rename
    # Find first wav in AUDIO_DIR that starts with our stem
    if not out_wav.exists():
        stem = out_wav.stem
        candidates = list(AUDIO_DIR.glob(f"{stem}*.wav"))
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            candidates[0].rename(out_wav)
    if not out_wav.exists():
        raise RuntimeError(f"Failed to produce {out_wav}")


def main() -> int:
    ensure_dirs()

    print("Enter inputs (leave optional fields blank):")
    url = input("Video URL: ").strip()
    title = input("Debate name (for filenames): ").strip()
    date = input("Date (YYYY-MM-DD): ").strip()
    length_raw = input("Length (HH:MM:SS or seconds) [optional]: ").strip()

    if not url:
        print("URL is required", file=sys.stderr)
        return 2
    base = slugify(title) if title else "debate"

    # Resolve paths
    wav_path = AUDIO_DIR / f"{base}.wav"
    txt_path = DEBATE_DIR / f"{base}.txt"
    debate_json_path = DEBATE_DIR / f"{base}.json"
    scored_json_path = SCORED_DIR / f"{base}.json"
    justify_json_path = JUSTIFY_DIR / f"{base}.json"

    # 1) Download audio
    end_hms = parse_duration(length_raw)
    print(f"→ Downloading audio to {wav_path}")
    download_audio(url, wav_path, end_hms)

    # 2) Prepare transcribe input (symlink to expected debate.wav)
    debate_link = AUDIO_DIR / "debate.wav"
    try:
        if debate_link.exists() or debate_link.is_symlink():
            debate_link.unlink()
        debate_link.symlink_to(wav_path.name)
    except Exception:
        # Fallback: copy
        shutil.copy2(wav_path, debate_link)

    # 3) Run transcribe.py (produces diatrize/debate_section.txt)
    print("→ Transcribing ...")
    run([sys.executable, str(ROOT / "transcribe.py")])

    # Move transcript to debate/<base>.txt
    src_txt = DIATRIZE / "debate_section.txt"
    if not src_txt.exists():
        raise RuntimeError(f"Expected transcript not found: {src_txt}")
    shutil.move(str(src_txt), str(txt_path))
    print(f"✓ Transcript saved -> {txt_path}")

    # 4) Run analyze_debate.py with metadata via stdin
    print("→ Analyzing transcript (segmenting to blocks) ...")
    meta_input = "\n".join([
        title,
        url,
        date,
        str(int(length_raw) if length_raw.isdigit() else "") if length_raw else "",
        "",
    ])
    run([sys.executable, str(DIATRIZE / "analyze_debate.py"), str(txt_path)], input_text=meta_input)

    # analyze writes diatrize/<stem>_analysis.json — rename to diatrize/debate/<base>.json
    analysis_src = DIATRIZE / f"{txt_path.stem}_analysis.json"
    if not analysis_src.exists():
        raise RuntimeError(f"Expected analysis JSON not found: {analysis_src}")
    shutil.move(str(analysis_src), str(debate_json_path))
    print(f"✓ Debate JSON -> {debate_json_path}")

    # 5) Run score_debate.py (non-interactive: pass input path and accept default output name)
    print("→ Scoring debate windows ...")
    # Accept default filename by sending an empty line
    run([sys.executable, str(DIATRIZE / "score_debate.py"), str(debate_json_path)], input_text="\n")
    if not scored_json_path.exists():
        # try to find a file with the stem in scored dir
        cand = list(SCORED_DIR.glob(f"{base}.json"))
        if cand:
            scored_json_path = cand[0]
    print(f"✓ Scored JSON -> {scored_json_path}")

    # 6) Run justify.py to produce per-window justification payload
    print("→ Creating justifications payload ...")
    run([sys.executable, str(DIATRIZE / "justify.py"), str(scored_json_path)])
    if not justify_json_path.exists():
        # fallback: find by stem
        cand = list(JUSTIFY_DIR.glob(f"{base}.json"))
        if cand:
            justify_json_path = cand[0]
    print(f"✓ Justify JSON -> {justify_json_path}")

    # Summary
    print("\nAll done. Outputs:")
    print("  Audio:", wav_path)
    print("  Transcript:", txt_path)
    print("  Debate JSON:", debate_json_path)
    print("  Scored JSON:", scored_json_path)
    print("  Justify JSON:", justify_json_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
        raise


