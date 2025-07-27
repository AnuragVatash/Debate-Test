#!/usr/bin/env python3
import argparse
import ffmpeg

def trim_wav(input_path: str, start: float, end: float, output_path: str):
    """
    Trim the input WAV file from `start` seconds to `end` seconds,
    copying the audio stream directly to avoid re-encoding.
    """
    (
        ffmpeg
        .input(input_path, ss=start, to=end)       # set start and end times:contentReference[oaicite:0]{index=0}  
        .output(output_path, acodec='copy')        # copy audio codec to preserve quality:contentReference[oaicite:1]{index=1}  
        .run(overwrite_output=True)                # execute and overwrite if exists:contentReference[oaicite:2]{index=2}  
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trim a WAV file using ffmpeg-python by specifying start/end times in seconds."
    )
    parser.add_argument("input",  help="Path to the input WAV file")  
    parser.add_argument("start",  type=float, help="Trim start time (seconds)")  
    parser.add_argument("end",    type=float, help="Trim end time (seconds)")  
    parser.add_argument("output", help="Path for the output trimmed WAV file")  
    args = parser.parse_args()

    trim_wav(args.input, args.start, args.end, args.output)
