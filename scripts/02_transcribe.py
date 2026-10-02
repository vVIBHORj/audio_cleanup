#!/usr/bin/env python3
"""
scripts/02_transcribe.py

Transcription stage for intercepted audio.
Loads output/denoised.wav, executes local speech-to-text inference
using faster-whisper, detects language automatically, formats output
with [HH:MM:SS.mmm] timestamps, handles uncertain speech,
and writes output/transcript.txt.
"""

import ctypes
import os
import re
import sys
from pathlib import Path

# Dependency check with clear installation messages
try:
    from faster_whisper import WhisperModel
    import soundfile as sf
except ImportError as err:
    sys.stderr.write(
        f"ERROR: Missing required dependency ({err.name}).\n"
        "Please install requirements using:\n"
        "    pip install -r requirements.txt\n"
    )
    sys.exit(1)

# ==============================================================================
# WHISPER CONFIGURATION
# ==============================================================================
MODEL_SIZE = "medium"  # Options: "tiny", "base", "small", "medium", "large-v3"
DEVICE = "auto"        # Options: "auto", "cuda", "cpu"
COMPUTE_TYPE = "auto"  # Options: "auto", "float16", "int8", "float32"
LANGUAGE = None        # Set to None for automatic language detection
# ==============================================================================


def check_cuda_support() -> bool:
    """Check if CUDA device is present and runtime libraries (cuBLAS) are accessible."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() <= 0:
            return False
        # Verify cublas runtime DLL is available on Windows
        if sys.platform == "win32":
            try:
                ctypes.CDLL("cublas64_12.dll")
            except OSError:
                return False
        return True
    except Exception:
        return False


def format_timestamp(seconds: float) -> str:
    """Format seconds into [HH:MM:SS.mmm] format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def clean_uncertain_or_hallucinated_speech(text: str) -> str:
    """
    Detect runaway repetitive loops or punctuation bursts caused by noise bursts
    and label with [unclear] to preserve uncertainty without fabricating speech.
    """
    text = re.sub(r"(?:\.\s*){5,}", "...", text).strip()
    words = text.split()
    if len(words) > 8:
        for n in range(1, 4):
            ngrams = [" ".join(words[i:i + n]).lower().strip(".,!?") for i in range(0, len(words) - n + 1, n)]
            if len(ngrams) > 4:
                most_common = max(set(ngrams), key=ngrams.count)
                if ngrams.count(most_common) / len(ngrams) > 0.60:
                    return "[unclear]"
    return text


def load_model(model_size: str, device_setting: str, compute_setting: str):
    """
    Load Whisper model with automatic CPU/GPU detection and optimal thread allocation.
    """
    if device_setting == "auto":
        is_cuda_ready = check_cuda_support()
        device = "cuda" if is_cuda_ready else "cpu"
    else:
        device = device_setting

    if compute_setting == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    else:
        compute_type = compute_setting

    cpu_threads = os.cpu_count() or 4
    if device == "cpu":
        print(f"Loading Whisper model '{model_size}' (device: cpu, compute_type: {compute_type}, threads: {cpu_threads})...")
        model = WhisperModel(model_size, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)
    else:
        try:
            print(f"Loading Whisper model '{model_size}' (device: {device}, compute_type: {compute_type})...")
            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as exc:
            print(f"Notice: Failed to load on {device} ({exc}). Falling back to CPU...")
            device = "cpu"
            compute_type = "int8"
            model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=cpu_threads)

    return model, device, compute_type


def transcribe_audio(audio_path: Path, output_path: Path) -> None:
    """
    Transcribe the denoised audio file and write timestamped transcript.
    """
    if not audio_path.exists():
        sys.stderr.write(
            "ERROR: output/denoised.wav was not found.\n"
            "Run 01_denoise.py first.\n"
        )
        sys.exit(1)

    # Verify audio file readability and duration
    try:
        info = sf.info(str(audio_path))
        print(f"Loaded audio: {audio_path.name} (Duration: {info.duration:.2f} s, Sample rate: {info.samplerate} Hz)")
    except Exception as exc:
        sys.stderr.write(f"ERROR: Failed to read audio file '{audio_path}': {exc}\n")
        sys.exit(1)

    model, active_device, active_compute = load_model(MODEL_SIZE, DEVICE, COMPUTE_TYPE)

    print("Transcribing speech...")
    try:
        segments_gen, trans_info = model.transcribe(
            str(audio_path),
            language=LANGUAGE,
            beam_size=5,
            condition_on_previous_text=False,
            repetition_penalty=1.15,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
            temperature=0.0,
            vad_filter=False,
        )
    except Exception as exc:
        sys.stderr.write(f"ERROR: Transcription failed: {exc}\n")
        sys.exit(1)

    lang_code = trans_info.language.upper()
    lang_prob = trans_info.language_probability
    print(f"Detected language: {lang_code} (probability: {lang_prob:.2%})")

    # Format transcript with timestamps
    segment_blocks = []
    print("Formatting transcript segments...")
    for segment in segments_gen:
        raw_text = segment.text.strip()
        if not raw_text:
            continue

        text = clean_uncertain_or_hallucinated_speech(raw_text)

        # If model indicates extremely low confidence, mark as unclear
        if segment.avg_logprob < -1.8 and len(text) < 4:
            text = "[unclear]"

        ts_start = format_timestamp(segment.start)
        formatted_entry = f"[{ts_start}] {text}"
        segment_blocks.append(formatted_entry)
        print(f"  {formatted_entry}")

    if not segment_blocks:
        segment_blocks.append("[No intelligible speech detected]")

    transcript_text = "\n\n".join(segment_blocks) + "\n"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_text(transcript_text, encoding="utf-8")
        print(f"\nTranscript saved to: {output_path} ({len(segment_blocks)} segments)")
    except Exception as exc:
        sys.stderr.write(f"ERROR: Failed to write transcript '{output_path}': {exc}\n")
        sys.exit(1)

    print("\nTranscription Summary:")
    print(f"  Audio duration: {info.duration:.2f} seconds")
    print(f"  Transcription language: {lang_code} ({lang_prob:.2%})")
    print(f"  Whisper model: {MODEL_SIZE}")
    print(f"  Processing device: {active_device} ({active_compute})")


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    if len(sys.argv) > 1:
        audio_path = Path(sys.argv[1])
    else:
        audio_path = project_root / "output" / "denoised.wav"

    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
    elif len(sys.argv) > 1 and audio_path.name != "denoised.wav":
        output_path = project_root / "output" / f"transcript_{audio_path.stem}.txt"
    else:
        output_path = project_root / "output" / "transcript.txt"

    transcribe_audio(audio_path, output_path)


if __name__ == "__main__":
    main()
