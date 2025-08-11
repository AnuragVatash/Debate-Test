# Debate Transcription with WhisperX

This project uses WhisperX for high-quality speech transcription with speaker diarization.

## Quick Start

### 1. Fix WhisperX Issues (Docker Container)

```bash
# Fix version issue (HTTP 301 errors)
./fix_whisperx_version.sh

# Fix VAD URL issue (if needed)
./fix_vad_github_issue.sh
```

### 2. Use Deepdml Model (Recommended)

```bash
# Setup and test
./setup_deepdml_docker.sh

# Transcribe with maximum performance
python3 transcribe_with_whisperx_ct2.py debate_section.wav --compute_type float16
```

## Files

### Core Scripts

- `transcribe_with_whisperx_ct2.py` - **Main script** using deepdml/faster-whisper-large-v3-turbo-ct2
- `transcribe_with_whisperx_cached.py` - Standard WhisperX with caching
- `transcribe_with_whisperx.py` - Basic WhisperX script

### Fix Scripts

- `fix_whisperx_version.sh` - Fixes HTTP 301 errors by installing WhisperX 3.2.0
- `fix_vad_github_issue.sh` - Fixes VAD model URL issues
- `setup_deepdml_docker.sh` - Complete setup for deepdml model

### Testing

- `test_deepdml_model.py` - Test the deepdml model

### Documentation

- `model_comparison.md` - Performance comparison of different models

### Original Files

- `transcribe.py` - Original faster-whisper script
- `trim_section.py` - Audio trimming utility
- `debate_section.wav` - Audio file to transcribe

## Model Options

1. **Deepdml CTranslate2** (Fastest): `deepdml/faster-whisper-large-v3-turbo-ct2`
2. **Standard Whisper**: `large-v3`
3. **Other models**: `tiny`, `base`, `small`, `medium`

## Usage Examples

```bash
# Maximum performance (recommended)
python3 transcribe_with_whisperx_ct2.py debate_section.wav --compute_type float16

# Standard with caching
python3 transcribe_with_whisperx_cached.py debate_section.wav --compute_type int8

# Skip diarization for speed
python3 transcribe_with_whisperx_ct2.py debate_section.wav --skip_diarization
```

## Output Files

- `debate_section.json` - Transcription with timestamps and speakers
- `debate_section.rttm` - Speaker diarization data
