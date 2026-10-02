#!/usr/bin/env python3
"""
scripts/01_denoise.py

Denoising stage for intercepted audio.
Loads input audio, performs inspection, conservative high-pass & spectral noise reduction,
speech-preserving bandpass filtering, and volume normalization.
Saves denoised audio to output/denoised.wav.
"""

import os
import sys
from pathlib import Path

# Dependency check with clear installation messages
try:
    import numpy as np
    import soundfile as sf
    import scipy.signal as signal
    import noisereduce as nr
except ImportError as err:
    sys.stderr.write(
        f"ERROR: Missing required dependency ({err.name}).\n"
        "Please install requirements using:\n"
        "    pip install -r requirements.txt\n"
    )
    sys.exit(1)


def find_input_file(project_root: Path) -> Path:
    """Locate the input audio file with fallback support."""
    # 1. Standard expected location: input/war-intercept.wav
    primary = project_root / "input" / "war-intercept.wav"
    if primary.is_file():
        return primary

    # 2. Fallback in project root: war-intercept.wav
    fallback = project_root / "war-intercept.wav"
    if fallback.is_file():
        return fallback

    # 3. Check current working directory
    cwd_input = Path.cwd() / "input" / "war-intercept.wav"
    if cwd_input.is_file():
        return cwd_input

    cwd_fallback = Path.cwd() / "war-intercept.wav"
    if cwd_fallback.is_file():
        return cwd_fallback

    return primary


def inspect_audio(file_path: Path, data: np.ndarray, sr: int, subtype: str) -> None:
    """Print audio metadata and statistics summary."""
    duration = len(data) / sr
    channels = "Mono" if data.ndim == 1 or data.shape[1] == 1 else "Stereo"
    rms = float(np.sqrt(np.mean(data**2)))
    peak = float(np.max(np.abs(data)))

    print("Audio information")
    print("------------------")
    print(f"File: {file_path.name}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: {sr} Hz")
    print(f"Channels: {channels}")
    if subtype:
        print(f"Bit depth: {subtype}")
    print(f"RMS: {rms:.5f}")
    print(f"Peak: {peak:.5f}")
    print("------------------")


def highpass_filter(data: np.ndarray, sr: int, cutoff_hz: float = 80.0, order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase high-pass Butterworth filter to remove low-frequency rumble,
    sub-audible thumps, and electrical mains hum below speech frequencies.
    """
    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, data)


def speech_band_filter(data: np.ndarray, sr: int, high_cutoff_hz: float = 7500.0, order: int = 4) -> np.ndarray:
    """
    Apply a gentle low-pass filter to attenuate extreme high-frequency hiss
    above intelligible human speech frequencies.
    """
    nyquist = sr / 2.0
    if high_cutoff_hz >= nyquist:
        return data
    sos = signal.butter(order, high_cutoff_hz, btype="lowpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, data)


def normalize_volume(data: np.ndarray, target_peak: float = 0.90) -> np.ndarray:
    """
    Safely normalize audio peak amplitude to prevent clipping and ensure consistent volume.
    """
    current_peak = np.max(np.abs(data))
    if current_peak > 1e-6:
        data = data * (target_peak / current_peak)
    return np.clip(data, -1.0, 1.0)


def denoise_audio(input_path: Path, output_path: Path) -> None:
    """
    Execute the speech-preserving denoising pipeline.
    """
    if not input_path.exists():
        sys.stderr.write("ERROR: input/war-intercept.wav was not found.\n")
        sys.exit(1)

    print(f"Loading input: {input_path}")
    try:
        data, sr = sf.read(str(input_path), dtype="float32")
        info = sf.info(str(input_path))
        subtype = info.subtype
    except Exception as exc:
        sys.stderr.write(f"ERROR: Failed to read audio file '{input_path}': {exc}\n")
        sys.exit(1)

    # Convert to mono if multi-channel
    if data.ndim > 1:
        print("Converting multi-channel audio to mono...")
        data = np.mean(data, axis=1)

    # Step 1: Inspection
    inspect_audio(input_path, data, sr, subtype)

    # Step 2: Denoising Pipeline
    print("Running denoising pipeline...")

    # A. High-pass filter (cuts sub-speech rumble & AC hum < 80 Hz)
    print("  [1/4] Applying conservative high-pass filter (80 Hz)...")
    filtered = highpass_filter(data, sr, cutoff_hz=80.0, order=4)

    # B. Moderate spectral noise reduction (preserving speech intelligibility)
    # Using conservative prop_decrease (0.70) to eliminate static without musical noise/distortion
    print("  [2/4] Applying moderate spectral noise reduction...")
    denoised = nr.reduce_noise(
        y=filtered,
        sr=sr,
        prop_decrease=0.70,
        stationary=True,
        n_fft=1024,
        win_length=1024,
        hop_length=256,
        n_std_thresh_stationary=1.5,
    )

    # C. Speech-preserving bandpass / gentle low-pass (7500 Hz cutoff)
    print("  [3/4] Applying speech-preserving filter...")
    refined = speech_band_filter(denoised, sr, high_cutoff_hz=7500.0, order=4)

    # D. Volume normalization
    print("  [4/4] Normalizing volume...")
    final_audio = normalize_volume(refined, target_peak=0.90)

    # Step 3: Save to output/denoised.wav
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving denoised audio to: {output_path}")
    try:
        sf.write(str(output_path), final_audio, sr, subtype="PCM_16")
    except Exception as exc:
        sys.stderr.write(f"ERROR: Failed to write output audio '{output_path}': {exc}\n")
        sys.exit(1)

    # Summary validation
    out_duration = len(final_audio) / sr
    out_peak = float(np.max(np.abs(final_audio)))
    out_rms = float(np.sqrt(np.mean(final_audio**2)))
    print("Denoising completed successfully.")
    print(f"  Output duration: {out_duration:.2f} s (Original: {len(data)/sr:.2f} s)")
    print(f"  Output RMS: {out_rms:.5f}, Peak: {out_peak:.5f}")


def main() -> None:
    # Project root is the parent directory of 'scripts'
    project_root = Path(__file__).resolve().parent.parent

    # Check CLI argument or search standard path
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        input_path = find_input_file(project_root)

    output_path = project_root / "output" / "denoised.wav"

    denoise_audio(input_path, output_path)


if __name__ == "__main__":
    main()
