#!/usr/bin/env python3
"""
run_pipeline.py

Executes the complete audio denoising and transcription pipeline:
    1. 01_denoise.py: Inspects and denoises input audio into output/denoised.wav
    2. 02_transcribe.py: Transcribes denoised audio into output/transcript.txt
"""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parent
    scripts_dir = project_root / "scripts"
    denoise_script = scripts_dir / "01_denoise.py"
    transcribe_script = scripts_dir / "02_transcribe.py"

    print("=" * 60)
    print("STEP 1: DENOISING AUDIO")
    print("=" * 60)
    ret = subprocess.run([sys.executable, str(denoise_script)])
    if ret.returncode != 0:
        sys.stderr.write(f"\nPipeline halted: Denoising failed with code {ret.returncode}.\n")
        sys.exit(ret.returncode)

    print("\n" + "=" * 60)
    print("STEP 2: TRANSCRIBING AUDIO")
    print("=" * 60)
    ret = subprocess.run([sys.executable, str(transcribe_script)])
    if ret.returncode != 0:
        sys.stderr.write(f"\nPipeline halted: Transcription failed with code {ret.returncode}.\n")
        sys.exit(ret.returncode)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
