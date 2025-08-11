#!/usr/bin/env python3
"""
Prepare minimal per-window payloads for LLM justification.

What this script does now:
- Loads the scored timeline (`diatrize/scored/<name>.json`) and matching debate blocks
  (`diatrize/debate/<name>.json`).
- Selects one highlight window per 5-minute bucket prioritizing windows where both A and B
  speak, else the window with the largest |delta|.
- Builds minimal payloads containing only necessary fields: identifiers, rubric_scores,
  block_range, and messages (speaker/name/text chunks) from the debate blocks. No full JSONs
  are embedded.
- Optionally calls an LLM (Gemini) to produce a 2–3 sentence justification that combines
  the excerpt and rubric signal for each selected window, if an API key is available.
- Writes the result to `diatrize/justify/<name>.json`.

CLI:
  python diatrize/justify.py <basename or path to scored json>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Optional: load environment variables from project root .env
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

if load_dotenv:
    try:
        _PROJECT_ROOT = Path(__file__).resolve().parents[1]
        load_dotenv(_PROJECT_ROOT / ".env")
    except Exception:
        pass

# Optional: Gemini client for LLM summarization
try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    genai = None  # type: ignore


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_hhmmss(ts: str | None) -> float:
    if not ts:
        return 0.0
    try:
        h, m, s = ts.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return 0.0


def _split_long_text(text: str, speaker: str, name: str, start: Any, end: Any) -> List[Dict[str, Any]]:
    words = (text or "").split()
    if len(words) < 130:
        return [{"speaker": speaker, "name": name, "text": text or "", "start": start, "end": end}]
    chunks: List[Dict[str, Any]] = []
    i = 0
    while i < len(words):
        j = min(i + 100, len(words))
        chunks.append({
            "speaker": speaker,
            "name": name,
            "text": " ".join(words[i:j]),
            "start": start,
            "end": end,
        })
        i = j
    return chunks


def select_highlight_windows(
    scored: Dict[str, Any],
    debate: Dict[str, Any],
    group_seconds: int = 300,
) -> List[Dict[str, Any]]:
    """
    Select one window per time bucket, preferring those where both A and B speak.

    Returns list of dicts with: group_idx, window_idx, start, end, delta, cumulative,
    block_range, rubric_scores, messages[].
    """
    timeline = scored.get("timeline") or scored.get("score_timeline") or []
    blocks_by_id: Dict[int, Dict[str, Any]] = {
        int(b["block_id"]): b
        for b in (debate.get("blocks") or [])
        if isinstance(b, dict) and "block_id" in b
    }
    speakers = debate.get("speakers", {})

    # Group windows into time buckets
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for it in timeline:
        start_value = it.get("start")
        if isinstance(start_value, str):
            start_s = _parse_hhmmss(start_value)
        else:
            # fallback: derive a rough second value from window index (1 min per window)
            start_s = float(it.get("window_idx", 0)) * 60.0
        g = int(start_s // group_seconds)
        groups.setdefault(g, []).append(it)

    def has_both_speakers(it: Dict[str, Any]) -> bool:
        br = it.get("block_range") or {}
        try:
            bmin = int(br.get("block_id_min"))
            bmax = int(br.get("block_id_max"))
        except Exception:
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

    selected: List[Dict[str, Any]] = []
    for gidx in sorted(groups):
        items = groups[gidx]
        both = [it for it in items if has_both_speakers(it)]
        pool = both if both else items
        best = max(pool, key=lambda x: abs(x.get("delta", 0) or 0))

        # Build messages from block_range
        br = best.get("block_range") or {}
        bmin = br.get("block_id_min")
        bmax = br.get("block_id_max")
        msgs: List[Dict[str, Any]] = []
        if bmin is not None and bmax is not None:
            try:
                for bid in range(int(bmin), int(bmax) + 1):
                    blk = blocks_by_id.get(int(bid))
                    if not blk:
                        continue
                    sp = blk.get("speaker")
                    if sp == "MOD":
                        continue
                    name_val = (speakers.get(sp, {}) or {}).get("name") or sp
                    msgs.extend(
                        _split_long_text(
                            blk.get("text", ""), sp, name_val, blk.get("start"), blk.get("end")
                        )
                    )
            except Exception:
                # be robust to malformed ranges
                pass

        selected.append({
            "group_idx": gidx,
            "window_idx": best.get("window_idx"),
            "start": best.get("start"),
            "end": best.get("end"),
            "delta": best.get("delta", 0),
            "cumulative": best.get("cumulative", 0),
            "block_range": br,
            "rubric_scores": best.get("rubric_scores"),
            "messages": msgs,
        })

    return selected


def _get_gemini_model():
    """Initialize and return a Gemini GenerativeModel if configured, else None.

    Uses env vars: GOOGLE_API_KEY or GEMINI_API_KEY. Model can be overridden with GEMINI_MODEL.
    """
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key or genai is None:
        return None, None
    model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        return model, model_name
    except Exception:
        return None, None


def _format_rubric_for_prompt(rubric_scores: Dict[str, Any], delta: Any) -> str:
    a = rubric_scores.get("A", {}) if isinstance(rubric_scores, dict) else {}
    b = rubric_scores.get("B", {}) if isinstance(rubric_scores, dict) else {}
    keys = ["clarity", "evidence", "rebuttal", "organization", "style", "total"]
    parts: List[str] = []
    for k in keys:
        av = a.get(k, "-")
        bv = b.get(k, "-")
        parts.append(f"{k}: A={av} B={bv}")
    return " | ".join(parts) + f" | Δ={delta}"


def _messages_to_excerpt(messages: List[Dict[str, Any]], max_chars: int = 3000) -> str:
    lines: List[str] = []
    used = 0
    for m in messages:
        text = (m.get("text") or "").strip()
        if not text:
            continue
        name = (m.get("name") or m.get("speaker") or "?")
        line = f"[{name}] {text}"
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def _summarize_with_llm(model: Any, window: Dict[str, Any]) -> str | None:
    if model is None:
        return None
    try:
        rubric = window.get("rubric_scores") or {}
        delta = window.get("delta")
        rubric_str = _format_rubric_for_prompt(rubric, delta)
        excerpt = _messages_to_excerpt(window.get("messages") or [], max_chars=3000)
        if not excerpt:
            return None
        prompt = (
            "You are a debate judge. Read the short excerpt and the rubric scores, then write a concise, "
            "2–3 sentence justification that explains which side appears stronger in this window and why. "
            "Ground your reasoning in both the scores and the content of the excerpt. Be neutral, precise, "
            "and avoid quoting large spans. Do not include headings or bullet points.\n\n"
            f"Rubric scores: {rubric_str}\n\n"
            "Excerpt (speaker-tagged):\n"
            f"{excerpt}\n\n"
            "Justification (2–3 sentences):"
        )
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", None)
        return (text or "").strip() or None
    except Exception:
        return None


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    scored_dir = project_root / "diatrize" / "scored"
    debate_dir = project_root / "diatrize" / "debate"
    justify_dir = project_root / "diatrize" / "justify"
    justify_dir.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) < 2:
        print("Usage: python diatrize/justify.py <basename or path to scored json>")
        return 2

    arg = Path(sys.argv[1])
    if arg.exists():
        scored_path = arg
    else:
        scored_path = scored_dir / arg.name

    if not scored_path.exists():
        print(f"Scored file not found: {scored_path}", file=sys.stderr)
        return 1

    name = scored_path.stem
    debate_path = debate_dir / f"{name}.json"
    if not debate_path.exists():
        print(f"Debate file not found for text: {debate_path}", file=sys.stderr)
        return 1

    scored = load_json(scored_path)
    debate = load_json(debate_path)

    selected = select_highlight_windows(scored, debate, group_seconds=300)

    # Optional LLM justification per selected window
    skip_llm = os.environ.get("JUSTIFY_SKIP_LLM", "0") == "1"
    model, model_name = (None, None) if skip_llm else _get_gemini_model()
    for w in selected:
        justification = _summarize_with_llm(model, w)
        if not justification:
            # Fallback: compact deterministic line from scores
            rubric = w.get("rubric_scores") or {}
            delta = w.get("delta")
            rubric_line = _format_rubric_for_prompt(rubric, delta)
            justification = (
                f"Based on the rubric ({rubric_line}), one side holds a marginal advantage in this window. "
                "The excerpt reflects the balance of claims and responses without clear dominance."
            )
        w["justification"] = justification

    out_payload: Dict[str, Any] = {
        "name": name,
        "selected_windows": selected,
        **({"llm_model": model_name} if model_name else {}),
    }

    out_path = justify_dir / f"{name}.json"
    out_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


