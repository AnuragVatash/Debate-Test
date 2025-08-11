#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, render_template, jsonify, abort
from flask_socketio import SocketIO


BASE_DIR = Path(__file__).resolve().parent.parent
DIATRIZE_DIR = BASE_DIR / "diatrize"
SCORED_DIR = DIATRIZE_DIR / "scored"
DEBATE_DIR = DIATRIZE_DIR / "debate"


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
    return app


app = create_app()
socketio = SocketIO(app, cors_allowed_origins="*")


def list_scored_files() -> List[Path]:
    if not SCORED_DIR.exists():
        return []
    return sorted(SCORED_DIR.glob("*.json"))


def _parse_hhmmss_to_seconds(ts: str) -> float:
    try:
        # Supports hh:mm:ss or hh:mm:ss.SSS
        hms, *_ = ts.split(" ")
        parts = hms.split(":")
        if len(parts) != 3:
            return 0.0
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s
    except Exception:
        return 0.0


def _load_analysis_for_scored(scored_path: Path) -> Dict[str, Any] | None:
    stem = scored_path.stem  # e.g., mydebate
    # Preferred: diatrize/debate/<stem>.json (full text JSON)
    if DEBATE_DIR.exists():
        candidate = DEBATE_DIR / f"{stem}.json"
        if candidate.exists():
            try:
                return load_json_file(candidate)
            except Exception:
                pass
    # Back-compat: diatrize/<stem>_analysis.json
    candidate = DIATRIZE_DIR / f"{stem}_analysis.json"
    if candidate.exists():
        try:
            return load_json_file(candidate)
        except Exception:
            pass
    return None


def _compute_highlights(scored: Dict[str, Any], analysis: Dict[str, Any] | None) -> Dict[str, Any]:
    timeline: List[Dict[str, Any]] = scored.get("timeline") or scored.get("score_timeline") or []
    if not isinstance(timeline, list):
        timeline = []

    # Map blocks by id from analysis for quick lookup
    blocks_by_id: Dict[int, Dict[str, Any]] = {}
    speakers: Dict[str, Any] = {}
    if analysis:
        for b in analysis.get("blocks", []) or []:
            try:
                blocks_by_id[int(b.get("block_id"))] = b
            except Exception:
                continue
        speakers = analysis.get("speakers", {})

    # Group windows into 5-minute bins by start time
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for item in timeline:
        start_s = 0.0
        if "start" in item and isinstance(item["start"], str):
            start_s = _parse_hhmmss_to_seconds(item["start"])
        elif "window_idx" in item:
            try:
                start_s = float(item["window_idx"]) * 60.0
            except Exception:
                start_s = 0.0
        group_idx = int(start_s // 300)
        groups.setdefault(group_idx, []).append(item)

    def split_text_bubbles(text: str, speaker: str, name: str, start: Any, end: Any) -> List[Dict[str, Any]]:
        # Split long blocks into multiple bubbles when length is large.
        # Policy: if total words >= 130, split into chunks of <= 100 words (target ~90).
        words = text.split()
        n = len(words)
        if n < 130:
            return [{
                "speaker": speaker,
                "name": name,
                "text": text,
                "start": start,
                "end": end,
            }]
        bubbles: List[Dict[str, Any]] = []
        max_chunk = 100
        target = 90
        i = 0
        while i < n:
            j = min(i + target, n)
            # ensure we don't exceed max_chunk
            j = min(j, i + max_chunk)
            chunk = " ".join(words[i:j])
            bubbles.append({
                "speaker": speaker,
                "name": name,
                "text": chunk,
                "start": start,
                "end": end,
            })
            i = j
        return bubbles

    # For each 5-min group pick the largest |delta|, prioritizing windows where both A and B spoke
    highlights: List[Dict[str, Any]] = []
    for gidx in sorted(groups.keys()):
        items = groups[gidx]
        if not items:
            continue
        def has_both_speakers(item: Dict[str, Any]) -> bool:
            br = item.get("block_range") or {}
            try:
                bmin = int(br.get("block_id_min")) if br else None
                bmax = int(br.get("block_id_max")) if br else None
            except Exception:
                return False
            if analysis is None or bmin is None or bmax is None:
                return False
            seen_a = False
            seen_b = False
            for bid in range(bmin, bmax + 1):
                blk = blocks_by_id.get(bid)
                if not blk:
                    continue
                spk = blk.get("speaker")
                if spk == "A":
                    seen_a = True
                elif spk == "B":
                    seen_b = True
                if seen_a and seen_b:
                    return True
            return False

        both_items = [it for it in items if has_both_speakers(it)]
        pool = both_items if both_items else items
        best = max(pool, key=lambda x: abs(x.get("delta", 0) or 0))
        # Build messages from block range, if available and analysis present
        messages: List[Dict[str, Any]] = []
        br = best.get("block_range") or {}
        try:
            bmin = int(br.get("block_id_min")) if br else None
            bmax = int(br.get("block_id_max")) if br else None
        except Exception:
            bmin = bmax = None
        if analysis and bmin is not None and bmax is not None:
            for bid in range(bmin, bmax + 1):
                blk = blocks_by_id.get(bid)
                if not blk:
                    continue
                spk = blk.get("speaker")
                if spk == "MOD":
                    continue
                name = speakers.get(spk, {}).get("name") or spk
                for bubble in split_text_bubbles(blk.get("text", ""), spk, name, blk.get("start"), blk.get("end")):
                    bubble["block_id"] = bid
                    messages.append(bubble)

        highlights.append({
            "group_idx": gidx,
            "window_idx": best.get("window_idx"),
            "start": best.get("start"),
            "end": best.get("end"),
            "delta": best.get("delta", 0),
            "cumulative": best.get("cumulative", 0),
            "block_range": br,
            "messages": messages,
            "rubric_scores": best.get("rubric_scores"),
        })

    # Compute min/max cumulative for eval bar scaling
    cumulatives = [float(x.get("cumulative", 0) or 0) for x in timeline]
    min_c = min(cumulatives) if cumulatives else -1.0
    max_c = max(cumulatives) if cumulatives else 1.0
    return {"highlights": highlights, "min_c": min_c, "max_c": max_c}


def load_json_file(fp: Path) -> Dict[str, Any]:
    try:
        text = fp.read_text(encoding="utf-8")
        return json.loads(text)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {fp}: {exc}") from exc


@app.get("/")
def index():
    debates = []
    for f in list_scored_files():
        try:
            data = load_json_file(f)
            title = data.get("metadata", {}).get("debate_title") or f.stem
            final = data.get("final_cumulative_score")
            debates.append({
                "id": f.stem,
                "file": f.name,
                "title": title,
                "final": final,
            })
        except Exception:
            debates.append({"id": f.stem, "file": f.name, "title": f.stem, "final": None})
    return render_template("index.html", debates=debates)


@app.get("/debate/<debate_id>")
def debate_view(debate_id: str):
    # Resolve file in diatrize
    candidates = [p for p in list_scored_files() if p.stem == debate_id]
    if not candidates:
        # Also allow passing a plain filename (without _scored)
        candidates = [p for p in list_scored_files() if p.stem.startswith(debate_id)]
    if not candidates:
        abort(404)
    fp = candidates[0]
    data = load_json_file(fp)
    # Provide minimal info to template for Plotly
    timeline = data.get("timeline", []) or data.get("score_timeline", [])
    speakers = data.get("speakers", {})
    analysis = _load_analysis_for_scored(fp)
    hl_info = _compute_highlights(data, analysis)
    return render_template(
        "debate.html",
        debate_id=debate_id,
        file_name=fp.name,
        metadata=data.get("metadata", {}),
        speakers=speakers,
        timeline=timeline,
        highlights=hl_info["highlights"],
        min_c=hl_info["min_c"],
        max_c=hl_info["max_c"],
        final=data.get("final_cumulative_score"),
        llm_declared_winner=data.get("llm_declared_winner"),
        a_wpm=data.get("a_wpm"),
        b_wpm=data.get("b_wpm"),
        confidence=data.get("confidence"),
    )


@app.get("/debate/<debate_id>/raw")
def debate_raw(debate_id: str):
    candidates = [p for p in list_scored_files() if p.stem == debate_id]
    if not candidates:
        abort(404)
    data = load_json_file(candidates[0])
    return jsonify(data)


@socketio.on("connect")
def on_connect():
    # Placeholder for future push updates
    pass


def main() -> None:
    # For dev only. In production use a proper WSGI/ASGI server.
    socketio.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)


if __name__ == "__main__":
    main()


