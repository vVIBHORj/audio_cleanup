#!/usr/bin/env python3
"""
scripts/transcribe_audio3.py

High-Precision Speech Transcription for output/audio3_test.wav.
Uses Whisper 'medium' (769M parameters, highest benchmark accuracy locally)
with optimal multi-threaded int8 inference, beam_size=5, condition_on_previous_text=False,
repetition_penalty=1.15, and strict hallucination filtering.

Outputs: output/transcript_audio3_test.txt
"""

import ctypes
import os
import re
import sys
import time
from pathlib import Path
import soundfile as sf
from faster_whisper import WhisperModel

AUDIO_INPUT = Path("output/audio3_test.wav")
TRANSCRIPT_OUTPUT = Path("output/transcript_audio3_test.txt")
MODEL_SIZE = "medium"


def check_cuda_support() -> bool:
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() <= 0:
            return False
        if sys.platform == "win32":
            try:
                ctypes.CDLL("cublas64_12.dll")
            except OSError:
                return False
        return True
    except Exception:
        return False


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def clean_uncertain_speech(text: str) -> str:
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


def load_model():
    is_cuda = check_cuda_support()
    device = "cuda" if is_cuda else "cpu"
    compute_type = "float16" if is_cuda else "int8"
    cpu_threads = os.cpu_count() or 8

    print(f"Loading Whisper model '{MODEL_SIZE}' (device: {device}, compute_type: {compute_type}, threads: {cpu_threads})...")
    model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type, cpu_threads=cpu_threads)
    return model, device, compute_type


def run_transcription():
    if not AUDIO_INPUT.exists():
        sys.stderr.write(f"ERROR: {AUDIO_INPUT} not found!\n")
        sys.exit(1)

    info = sf.info(str(AUDIO_INPUT))
    print(f"\nSource: {AUDIO_INPUT} ({info.duration:.2f}s, {info.samplerate} Hz, {info.channels} channel)")

    t0 = time.time()
    model, device, compute_type = load_model()
    t_load = time.time() - t0
    print(f"Model ready in {t_load:.2f}s.\n")

    print("Running transcription with optimal decoding parameters...")
    print("  - Model: medium (769M parameters)")
    print("  - Beam size: 5")
    print("  - Temperature: 0.0")
    print("  - Condition on previous text: False")
    print("  - Repetition penalty: 1.15")
    print("  - Compression ratio threshold: 2.4")
    print("  - No speech threshold: 0.6\n")

    t_trans0 = time.time()
    segments_gen, trans_info = model.transcribe(
        str(AUDIO_INPUT),
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        repetition_penalty=1.15,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.6,
        vad_filter=False,
    )

    lang_code = trans_info.language.upper()
    lang_prob = trans_info.language_probability
    print(f"Detected language: {lang_code} (probability: {lang_prob:.2%})\n")

    segment_lines = []
    for seg in segments_gen:
        raw_text = seg.text.strip()
        if not raw_text:
            continue

        clean_text = clean_uncertain_speech(raw_text)
        if seg.avg_logprob < -1.8 and len(clean_text) < 4:
            clean_text = "[unclear]"

        ts = format_timestamp(seg.start)
        line = f"[{ts}] {clean_text}"
        segment_lines.append(line)
        print(f"  {line}", flush=True)

    if not segment_lines:
        segment_lines.append("[No intelligible speech detected]")

    transcription_time = time.time() - t_trans0
    rtf = transcription_time / info.duration

    final_content = "\n\n".join(segment_lines) + "\n"
    TRANSCRIPT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_OUTPUT.write_text(final_content, encoding="utf-8")

    print(f"\n" + "=" * 60)
    print("TRANSCRIPTION SUMMARY")
    print("=" * 60)
    print(f"Input file:          {AUDIO_INPUT}")
    print(f"Output file:         {TRANSCRIPT_OUTPUT}")
    print(f"Model:               {MODEL_SIZE} (769M params)")
    print(f"Device:              {device} ({compute_type})")
    print(f"Audio duration:      {info.duration:.2f} s")
    print(f"Inference time:      {transcription_time:.2f} s ({rtf:.3f}x real-time)")
    print(f"Segments produced:   {len(segment_lines)}")
    print(f"Language:            {lang_code} ({lang_prob:.2%})")
    print("=" * 60)


if __name__ == "__main__":
    run_transcription()
