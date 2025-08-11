#!/usr/bin/env python3
"""
Unified WhisperX transcription script: ASR + alignment + (optional) speaker diarisation.
Supports:
- Any Whisper or CTranslate2 model (deepdml, openai, etc.)
- All compute types (float16, int8, etc.)
- Skipping diarization for speed
- Local cache setup
- Handles VAD-disabled environments gracefully

Usage:
  python transcribe_whisperx_unified.py debate_section.wav --model deepdml/faster-whisper-large-v3-turbo-ct2 --compute_type float16
"""
import argparse, json, os, gc, torch
from faster_whisper import WhisperModel
from pathlib import Path

def setup_cache():
    """Setup model cache directories"""
    cache_dir = Path.home() / ".cache" / "whisperx"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    print(f"📁 Using cache directory: {cache_dir}")
    return cache_dir

def main():
    parser = argparse.ArgumentParser(
        description="Unified WhisperX transcription + diarisation script")
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument("--model", default="deepdml/faster-whisper-large-v3-turbo-ct2",
                        help="Model name (Whisper, CTranslate2, deepdml, etc.)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--compute_type", default="float16",
                        help='float16|int8 (set int8 for CPU)')
    parser.add_argument("--out_json", help="Output JSON filename (default: <audio>.json)")
    parser.add_argument("--skip_diarization", action="store_true",
                        help="Skip speaker diarization to save time")
    args = parser.parse_args()

    # Setup cache
    cache_dir = setup_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_fp = args.audio
    out_json = args.out_json or f"{os.path.splitext(audio_fp)[0]}.json"

    # Determine compute type based on device
    requested_compute_type = args.compute_type
    compute_type = requested_compute_type
    if device == "cpu" and requested_compute_type.lower() in {"float16", "float32", "bfloat16"}:
        print("⚠️  CPU detected; overriding compute_type to int8 for compatibility")
        compute_type = "int8"

    print(f"🚀 Starting WhisperX transcription...")
    print(f"📱 Device: {device}")
    print(f"🎵 Audio: {audio_fp}")
    print(f"💾 Cache: {cache_dir}")
    print(f"⚡ Model: {args.model}")

    try:
        # 1) Transcribe (via faster-whisper to avoid WhisperX VAD path) -----------------
        print("\n📝 Step 1: Transcribing audio...")
        # Use faster-whisper directly; disables VAD to avoid any model download
        fw_model = WhisperModel(args.model, device=device, compute_type=compute_type)
        segments_iter, info = fw_model.transcribe(
            audio_fp,
            batch_size=args.batch_size,
            vad_filter=False,
            word_timestamps=True,
        )
        # Collect segments into WhisperX-compatible structure
        collected_segments = []
        for seg in segments_iter:
            segment_dict = {
                "start": float(seg.start) if seg.start is not None else 0.0,
                "end": float(seg.end) if seg.end is not None else 0.0,
                "text": seg.text or "",
            }
            if getattr(seg, "words", None):
                segment_dict["words"] = [
                    {
                        "start": float(w.start) if w.start is not None else 0.0,
                        "end": float(w.end) if w.end is not None else 0.0,
                        "word": w.word,
                    }
                    for w in seg.words
                    if w is not None
                ]
            collected_segments.append(segment_dict)

        result = {
            "language": info.language or "unknown",
            "segments": collected_segments,
        }
        del fw_model; gc.collect()
        print(f"✅ Transcription complete. Language detected: {result['language']}")

        # 2) Skip alignment (already have word timestamps) ------------------------------
        print("\n⏭️  Skipping explicit alignment (word timestamps already computed)")

        # 3) Speaker diarisation (optional) --------------------------------------------
        if not args.skip_diarization:
            print("\n👥 Step 3: Speaker diarization...")
            try:
                import whisperx  # lazy import; only needed for diarization utilities
                diar_model = whisperx.diarize.DiarizationPipeline(
                    use_auth_token=os.getenv("HF_TOKEN"), device=device)
                # Load audio via WhisperX utility (float32 mono 16k)
                audio = whisperx.load_audio(audio_fp)
                dia_segments = diar_model(audio)
                result = whisperx.assign_word_speakers(dia_segments, result)
                print("✅ Speaker diarization complete")
            except Exception as e:
                print(f"⚠️  Diarization failed: {e}")
                dia_segments = None
        else:
            print("\n⏭️  Skipping speaker diarization...")
            dia_segments = None

        # 4) Save outputs ---------------------------------------------------------------
        print(f"\n💾 Step 4: Saving results...")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"language": result["language"],
                       "segments": result["segments"]}, f, indent=2)
        print(f"✅ JSON saved: {out_json}")
        if dia_segments is not None:
            rttm_file = os.path.splitext(audio_fp)[0] + ".rttm"
            with open(rttm_file, "w") as f:
                dia_segments.write_rttm(f)
            print(f"✅ RTTM saved: {rttm_file}")

        # Print summary
        print(f"\n🎉 Processing complete!")
        print(f"📊 Summary:")
        print(f"   - Audio duration: {len(audio)/16000:.1f} seconds")
        print(f"   - Segments: {len(result['segments'])}")
        print(f"   - Language: {result['language']}")
        if dia_segments is not None:
            print(f"   - Speakers detected: {len(set(seg.get('speaker', 'UNKNOWN') for seg in result['segments']))}")

    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main()) 