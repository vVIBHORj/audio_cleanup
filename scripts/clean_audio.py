#!/usr/bin/env python3
"""
scripts/clean_audio.py

Applies targeted frequency filtering to remove useless low/high bands
and piercing carrier whistles, outputting final_clean_audio.wav.
"""

from pathlib import Path
import numpy as np
import scipy.signal as signal
import soundfile as sf
import noisereduce as nr


def clean_audio(input_file: Path, output_file: Path):
    data, sr = sf.read(str(input_file), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # 1. Bandpass filter: 250 Hz - 3800 Hz (cuts sub-rumble & high-pitch screech > 3.8 kHz)
    sos = signal.butter(4, [250, 3800], btype="bandpass", fs=sr, output="sos")
    filtered = signal.sosfiltfilt(sos, data)

    # 2. Notch filters: eradicate persistent 2601.6 Hz carrier and 1398.4 Hz harmonic whistles
    for freq in [2601.6, 1398.4]:
        b, a = signal.iirnotch(freq, Q=30, fs=sr)
        filtered = signal.filtfilt(b, a, filtered)

    # 3. Spectral noise reduction for residual in-band static
    denoised = nr.reduce_noise(y=filtered, sr=sr, prop_decrease=0.75, stationary=True)

    # 4. Volume normalization
    peak = np.max(np.abs(denoised))
    if peak > 1e-6:
        denoised = (denoised / peak) * 0.90

    output_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_file), denoised, sr, subtype="PCM_16")

    # Also save a copy in project root for convenience
    root_copy = output_file.parent.parent / "final_clean_audio.wav"
    sf.write(str(root_copy), denoised, sr, subtype="PCM_16")

    print(f"Created: {output_file} ({len(denoised)/sr:.2f}s)")
    print(f"Created: {root_copy}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    src = root / "input" / "war-intercept.wav"
    if not src.exists():
        src = root / "war-intercept.wav"
    dst = root / "output" / "final_clean_audio.wav"
    clean_audio(src, dst)
