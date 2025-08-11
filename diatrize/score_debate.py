#!/usr/bin/env python3
"""
Score a diarized debate JSON using an Elo-style rolling evaluation with Gemini.

Input: diatrize/debate_section_analysis.json (or a provided path)
Output: diatrize/debate_section_scored.json

Reads .env for GEMINI_API_KEY and uses model gemini-2.5-pro by default.
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a meticulous debate adjudication engine. \n"
    "Your job is to read a single JSON file that contains:\n"
    " \u2022 metadata about the debate,\n"
    " \u2022 a \"speakers\" dictionary keyed by short IDs (A, B, MOD, \u2026),\n"
    " \u2022 an ordered \"blocks\" array (each block = one contiguous speaking turn),\n"
    " \u2022 a summary object with raw word counts and speaking-time seconds.\n\n"
    "After loading, you will:\n"
    " 1. Slice the debate into rolling 60-second evaluation windows.  \n"
    "    \u2022 Window n spans [n \u00d7 60 s, (n + 1) \u00d7 60 s).  \n"
    "    \u2022 A window may contain multiple blocks or one long block; include every word whose timestamp lies in the window.  \n"
    " 2. For each window(ignore moderators), judge which debater performed better using the five-factor rubric below.  \n"
    " 3. Convert that judgment into an Elo delta (\u0394).  \n"
    "    Start the match score at 0.  \n"
    "    \u2022 Positive \u0394 means Speaker A gained ground.  \n"
    "    \u2022 Negative \u0394 means Speaker B gained ground.  \n"
    "    \u2022 Use K = 2.0 for normal windows, K = 1.0 if the window is mostly small-talk (fewer than 20 words total), and K = 3.5 if the window contains a direct clash (both sides rebutting the same claim).  \n"
    "    \u2022 Compute expected score E via logistic:  \n"
    "         EA = 1 / (1 + 10^( (SB \u2212 SA) / 400 ) )  \n"
    "      but because this is a two-player zero-sum match starting at SA = SB = 0, you can simplify:  \n"
    "         \u0394 = K \u00d7 (S \u2212 E) where  \n"
    "           S = 1 if A clearly wins the window,  \n"
    "           S = 0.5 if it is roughly tied,  \n"
    "           S = 0 if B clearly wins.\n"
    "    \u2022 Round \u0394 to 2 dp for readability and add it to the cumulative score.  \n"
    " 4. When all windows are processed, output:  \n"
    "    \u2022 timeline \u2013 array of { window_idx, start, end, block_range, rubric_scores, delta, cumulative }.  \n"
    "    \u2022 final_cumulative_score.  \n"
    "    \u2022 llm_declared_winner (A if score > 0, B if score < 0, \"Tied\" if == 0).  \n"
    "    \u2022 confidence from 0\u20131 (your subjective certainty about the winner).  \n"
    "    \u2022 a_wpm, b_wpm computed from the raw word / speaking-time stats.  \n"
    "    \u2022 A short natural-language verdict (<= 4 sentences) explaining why the winner prevailed overall.\n\n"
    "For each timeline entry: \n"
    "  - Provide block_range with only IDs: { block_id_min, block_id_max } for debater blocks overlapping the window (exclude moderators).\n"
    "  - Do NOT include any raw transcript text anywhere in the output.\n"
    "  - Provide rubric_scores with per-category scores in [0,1] for each debater and their totals, e.g.:\n"
    "    rubric_scores: {\n"
    "      A: { clarity: number, evidence: number, rebuttal: number, organization: number, style: number, total: number },\n"
    "      B: { clarity: number, evidence: number, rebuttal: number, organization: number, style: number, total: number }\n"
    "    }\n"
    "Scoring math for delta: Let SA be A's total (0..5) and SB be B's total (0..5).\n"
    "Compute normalized_diff = (SA - SB) / 5.0. Then \u0394 = K * normalized_diff with K rules unchanged.\n"
    "Use strict JSON and avoid extra commentary.\n\n"
    "Rubric for judging each 60-second window\n"
    "Score each debater 0\u20135 on the five factors, then compare totals:\n\n"
    "1. Clarity & Accuracy \u2013 presents ideas clearly, facts correct.  \n"
    "2. Evidence & Examples \u2013 uses data, anecdotes, citations.  \n"
    "3. Rebuttal & Responsiveness \u2013 addresses the opponent\u2019s points.  \n"
    "4. Organization & Logic \u2013 arguments flow, conclusions follow.  \n"
    "5. Style / Persuasion \u2013 confidence, rhetoric, tone, engagement.\n\n"
    "The debater with the higher total wins the window.  \n"
    "If totals differ by \u2264 1 point, call the window a tie (S = 0.5).\n\n"
    "Important: Judge only what is said inside the current window.  \n"
    "Do not let earlier or later statements sway that specific judgment.\n\n"
    "Return your result as a single JSON object \u2013 no extra prose outside JSON."
)


def load_env()  -> None:
    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:
        # dotenv is optional; skip if not installed
        return
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")


def main() -> int:
    load_env()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is missing in .env", file=sys.stderr)
        return 1

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

    # Resolve inputs/outputs
    project_root = Path(__file__).resolve().parents[1]
    debate_dir = project_root / "diatrize" / "debate"
    debate_dir.mkdir(parents=True, exist_ok=True)

    # Determine input file: CLI arg, or interactive search in diatrize/debate
    input_path: Path | None = None
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1])
        if not candidate.is_absolute():
            # try relative to debate_dir
            rel = debate_dir / candidate.name
            if rel.exists():
                input_path = rel
            else:
                input_path = candidate.resolve()
        else:
            input_path = candidate
    else:
        files = sorted(debate_dir.glob("*.json"))
        if not files:
            print(f"No debate JSON files in {debate_dir}", file=sys.stderr)
            return 1
        print("Available debates:")
        for idx, fp in enumerate(files, 1):
            print(f"  {idx}. {fp.name}")
        sel = input("Type a filename or number: ").strip()
        if sel.isdigit():
            i = int(sel)
            if 1 <= i <= len(files):
                input_path = files[i-1]
        if input_path is None:
            # fuzzy match by substring
            lc = sel.lower()
            matches = [fp for fp in files if lc in fp.name.lower()]
            if len(matches) == 1:
                input_path = matches[0]
            elif len(matches) > 1:
                print("Multiple matches:")
                for idx, fp in enumerate(matches, 1):
                    print(f"  {idx}. {fp.name}")
                sel2 = input("Choose number: ").strip()
                if sel2.isdigit():
                    j = int(sel2)
                    if 1 <= j <= len(matches):
                        input_path = matches[j-1]

    if input_path is None or not input_path.exists():
        print(f"Input file not found or not selected.", file=sys.stderr)
        return 1

    # Save scored outputs to diatrize/scored (single folder, per user preference)
    output_dir = project_root / "diatrize" / "scored"
    output_dir.mkdir(parents=True, exist_ok=True)
    default_name = f"{input_path.stem.replace('_analysis','')}.json"
    try:
        user_name = input(f"Output filename (saved in diatrize/) [{default_name}]: ").strip()
    except EOFError:
        user_name = ""
    if not user_name:
        user_name = default_name
    if not user_name.lower().endswith(".json"):
        user_name += ".json"
    # Sanitize path traversal by taking only the name
    user_name = Path(user_name).name
    output_path = output_dir / user_name

    # Build the two-part prompt
    try:
        debate_json = input_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Failed to read input: {e}", file=sys.stderr)
        return 1

    user_prompt = (
        "Here is the debate transcript in JSON format.\n"
        "Carry out the scoring procedure described in the system message:\n\n"
        f"{debate_json}"
    )

    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        print("google-generativeai not installed. Run: pip install -r diatrize/requirements.txt", file=sys.stderr)
        return 1

    genai.configure(api_key=api_key)

    # Keep JSON output if supported
    gen_cfg = {"temperature": 0.2, "response_mime_type": "application/json"}

    try:
        model = genai.GenerativeModel(model_name, generation_config=gen_cfg, system_instruction=SYSTEM_PROMPT)
    except TypeError:
        model = genai.GenerativeModel(model_name)

    def _response_to_text(resp: object) -> str:
        out: list[str] = []
        for c in getattr(resp, "candidates", []) or []:
            content = getattr(c, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if parts:
                for p in parts:
                    t = getattr(p, "text", None)
                    if t:
                        out.append(t)
        return ("".join(out)).strip()

    try:
        response = model.generate_content(user_prompt)
        text = _response_to_text(response)
        if not text:
            # Fallback without JSON mime type
            try:
                plain_model = genai.GenerativeModel(model_name, generation_config={"temperature": 0.2})
                response2 = plain_model.generate_content(user_prompt + "\n\nReturn ONLY strict JSON.")
                text = _response_to_text(response2)
            except Exception:
                text = ""
        if not text:
            raise RuntimeError("Empty response from model")
        output_path.write_text(text, encoding="utf-8")
    except Exception as e:
        print(f"Gemini call failed: {e}", file=sys.stderr)
        return 1

    # Normalize to pretty JSON if possible
    try:
        data = json.loads(text)
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # Leave raw text if not strict JSON
        pass

    print(f"✅ Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


