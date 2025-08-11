#!/usr/bin/env python3
"""
Analyze a debate transcript with Gemini and save a JSON summary.

Reads the transcript from debate_section.txt (or a provided path) and asks the
model for structured JSON output. API key is read from .env as GEMINI_API_KEY.

Usage:
  python diatrize/analyze_debate.py [optional_path_to_transcript]

Environment:
  - .env at project root containing GEMINI_API_KEY=...
  - Optional: GEMINI_MODEL to override model (default: gemini-2.5-pro)
"""
from __future__ import annotations

import os
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple


def load_env_from_project_root() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        # dotenv is optional; skip if not installed
        return
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")


def read_text_file(path: Path) -> str:
    if not path.exists():
        print(f"Input file not found: {path}", file=sys.stderr)
        sys.exit(1)
    return path.read_text(encoding="utf-8")


def build_prompt(transcript_text: str, metadata: dict, part_idx: int | None = None, total_parts: int | None = None) -> str:
    system = (
        "You are an expert debate transcriber and data-formatter.\n"
        "Your job is to take a raw timestamped transcript that contains NO speaker labels\n"
        "and convert it into a clean, machine-readable JSON object.\n"
    )

    principles = (
        "CORE PRINCIPLES\n"
        "1. A speaker change happens only when the content, tone, or conversational context clearly shifts from one person to another.\n"
        "2. A pause or silence is NOT sufficient evidence of a new speaker.\n"
        "   • Short filler noises (\"uh\", \"um\"), breath sounds, or timestamp gaps can occur within the same speaker’s turn.\n"
        "3. When uncertain, read several lines ahead (look-ahead) and behind (look-back) before assigning a speaker label.\n"
        "   • Examine diction, register, recurring phrases, stance, and rhetorical style to match utterances to the correct speaker.\n"
        "4. If you still cannot decide after context inspection, label with \"UNK\" but keep the block; do NOT split text sentence-by-sentence.\n"
    )

    output_req = (
        "OUTPUT REQUIREMENTS\n"
        "Return ONLY valid JSON in the exact structure below.\n"
        "Leave unknown fields empty (\"\", {}, [], or null).\n"
        "Timestamps must remain in hh:mm:ss.SSS format.\n\n"
        "{\n"
        "  \"metadata\": {\n"
        "    \"debate_title\": \"\",\n"
        "    \"video_url\": \"\",\n"
        "    \"date\": \"\",\n"
        "    \"duration_sec\": null\n"
        "  },\n"
        "  \"speakers\": {\n"
        "    \"A\":   { \"name\": \"\", \"role\": \"debater\",   \"bio\": \"\", \"affiliation\": \"\", \"external_links\": [], \"static_attributes\": {}, \"llm_profile\": {} },\n"
        "    \"B\":   { \"name\": \"\", \"role\": \"debater\",   \"bio\": \"\", \"affiliation\": \"\", \"external_links\": [], \"static_attributes\": {}, \"llm_profile\": {} },\n"
        "    \"MOD\": { \"name\": \"\", \"role\": \"moderator\", \"bio\": \"\", \"affiliation\": \"\", \"external_links\": [], \"static_attributes\": {}, \"llm_profile\": {} }\n"
        "  },\n"
        "  \"blocks\": [\n"
        "    {\n"
        "      \"block_id\": 1,\n"
        "      \"speaker\": \"B\",\n"
        "      \"start\": \"00:00:00.000\",\n"
        "      \"end\":   \"00:00:04.240\",\n"
        "      \"text\": \"Don't worry, I'm here.\"\n"
        "    }\n"
        "  ],\n"
        "  \"summary\": {\n"
        "    \"total_blocks\": 0,\n"
        "    \"word_counts\":   { \"A\": 0, \"B\": 0, \"MOD\": 0, \"UNK\": 0 },\n"
        "    \"speaking_time\": { \"A\": 0, \"B\": 0, \"MOD\": 0, \"UNK\": 0 }\n"
        "  }\n"
        "}\n\n"
        "Return ONLY the JSON object. Do not include any commentary.\n"
    )

    provided_meta = json.dumps({
        "metadata": metadata
    }, ensure_ascii=False)

    header = ""
    if part_idx is not None and total_parts is not None:
        header = f"THIS IS PART {part_idx+1} OF {total_parts}. Process only the lines below for this part. Return strictly valid JSON for this part. The blocks you output must be in chronological order for this part. Do not include any text outside these lines.\n\n"

    return (
        system + "\n\n" + principles + "\n\n" + output_req +
        "Provided metadata (use these values to fill the metadata block):\n" +
        provided_meta + "\n\n" +
        header +
        "Raw transcript lines (tab-separated: start\tend\ttext):\n\n" + transcript_text
    )


def main() -> int:
    load_env_from_project_root()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is missing. Put it in .env at project root.", file=sys.stderr)
        return 1

    # Model selection with sensible default; user requested gemini 2.5 pro
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

    # Resolve project root and default input (from diatrize/debate/)
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "diatrize" / "debate" / "debate_section.txt"
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1]).resolve()

    transcript_text = read_text_file(input_path)

    # Interactive metadata collection
    print("Enter metadata values. Leave blank to keep empty.")
    title = input("debate_title: ").strip()
    video_url = input("video_url: ").strip()
    date = input("date (YYYY-MM-DD): ").strip()
    duration_raw = input("duration_sec (integer seconds): ").strip()
    duration_val = None
    if duration_raw:
        try:
            duration_val = int(duration_raw)
        except ValueError:
            print("Invalid duration; leaving null.")
            duration_val = None

    metadata = {
        "debate_title": title,
        "video_url": video_url,
        "date": date,
        "duration_sec": duration_val,
    }

    # Split transcript into chunks to avoid long requests
    lines = [ln for ln in transcript_text.splitlines() if ln.strip()]
    # Heuristic chunk count; smaller chunks reduce 504s/empty parts
    total_parts = 6 if len(lines) >= 1200 else 5 if len(lines) >= 600 else 4 if len(lines) >= 240 else 2 if len(lines) >= 120 else 1
    if total_parts == 1:
        chunks = ["\n".join(lines)]
    else:
        size = max(1, (len(lines) + total_parts - 1) // total_parts)
        chunks = ["\n".join(lines[i:i+size]) for i in range(0, len(lines), size)]
        chunks = chunks[:total_parts]

    # Prepare output path
    output_dir = project_root / "diatrize"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_analysis.json"

    # Call Gemini (in parallel per chunk)
    try:
        import google.generativeai as genai  # type: ignore
    except Exception as e:
        print(
            "google-generativeai not installed. Run: pip install -r diatrize/requirements.txt",
            file=sys.stderr,
        )
        return 1

    genai.configure(api_key=api_key)

    # Prefer JSON responses if supported by the SDK version
    generation_config = {
        "temperature": 0.2,
        "response_mime_type": "application/json",
    }

    try:
        model = genai.GenerativeModel(
            model_name,
            generation_config=generation_config,
            system_instruction=(
                "You are an expert debate transcriber and data-formatter. Convert raw time"
                "stamped text with no speaker labels into a clean machine-readable JSON."
            ),
        )
    except TypeError:
        model = genai.GenerativeModel(model_name)

    def _response_to_text(resp: Any) -> str:
        # Avoid resp.text quick accessor; aggregate parts text from candidates
        out: List[str] = []
        for c in getattr(resp, "candidates", []) or []:
            content = getattr(c, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        out.append(t)
        return ("".join(out)).strip()

    def call_part(idx_chunk: Tuple[int, str]) -> Tuple[int, str | None]:
        idx, chunk_text = idx_chunk
        try:
            part_prompt = build_prompt(chunk_text, metadata, part_idx=idx, total_parts=len(chunks))
            resp = model.generate_content(part_prompt)
            txt = _response_to_text(resp)
            if not txt:
                # Fallback retry without JSON mime type to bypass quick accessor issues
                try:
                    fallback_model = genai.GenerativeModel(model_name, generation_config={"temperature": 0.2})
                    resp2 = fallback_model.generate_content(part_prompt + "\n\nReturn ONLY strict JSON.")
                    txt = _response_to_text(resp2)
                except Exception:
                    txt = ""
            return (idx, txt or None)
        except Exception as e:
            print(f"Part {idx+1} failed: {e}", file=sys.stderr)
            return (idx, None)

    futures = []
    # Limit concurrency to reduce rate/timeouts
    with ThreadPoolExecutor(max_workers=min(3, len(chunks))) as ex:
        for idx, chunk in enumerate(chunks):
            futures.append(ex.submit(call_part, (idx, chunk)))

    part_texts: Dict[int, str] = {}
    for fut in as_completed(futures):
        idx, txt = fut.result()
        if txt:
            part_texts[idx] = txt

    if not part_texts:
        print("All parts failed", file=sys.stderr)
        return 1

    # Parse parts and merge blocks
    def parse_json_maybe(s: str) -> Dict[str, Any] | None:
        try:
            return json.loads(s)
        except Exception:
            return None

    all_blocks: List[Dict[str, Any]] = []
    merged_speakers: Dict[str, Any] = {}
    for idx in sorted(part_texts.keys()):
        parsed = parse_json_maybe(part_texts[idx])
        if not parsed:
            continue
        # Merge speakers (first non-empty wins per field)
        sp = parsed.get("speakers") or {}
        for k, v in sp.items():
            if k not in merged_speakers:
                merged_speakers[k] = v
            else:
                # Fill missing fields
                for fk, fv in v.items():
                    if not merged_speakers[k].get(fk) and fv:
                        merged_speakers[k][fk] = fv
        # Extend blocks
        blks = parsed.get("blocks") or []
        for b in blks:
            if isinstance(b, dict):
                all_blocks.append(b)

    # Sort blocks by start time and renumber block_id
    def to_seconds(ts: str | None) -> float:
        if not ts:
            return 0.0
        try:
            h, m, s = ts.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
        except Exception:
            return 0.0

    all_blocks.sort(key=lambda b: to_seconds(b.get("start")))
    for i, b in enumerate(all_blocks, 1):
        b["block_id"] = i

    # Recompute summary
    def word_count(text: str | None) -> int:
        return len((text or "").split())

    summary = {
        "total_blocks": len(all_blocks),
        "word_counts": {"A": 0, "B": 0, "MOD": 0, "UNK": 0},
        "speaking_time": {"A": 0, "B": 0, "MOD": 0, "UNK": 0},
    }
    for b in all_blocks:
        sp = (b.get("speaker") or "UNK")
        if sp not in summary["word_counts"]:
            summary["word_counts"][sp] = 0
            summary["speaking_time"][sp] = 0
        summary["word_counts"][sp] += word_count(b.get("text"))
        summary["speaking_time"][sp] += max(0.0, to_seconds(b.get("end")) - to_seconds(b.get("start")))

    final_json = {
        "metadata": metadata,
        "speakers": merged_speakers,
        "blocks": all_blocks,
        "summary": summary,
    }

    output_path.write_text(json.dumps(final_json, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


