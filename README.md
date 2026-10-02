# Intercepted Audio Denoising & Transcription Pipeline

A minimal, reliable local Python pipeline that denoises intercepted audio and generates timestamped speech-to-text transcriptions using `noisereduce` and `faster-whisper`.

---

## Workflow

```text
input/war-intercept.wav
         ↓
scripts/01_denoise.py (Inspection, High-pass, Conservative Spectral Denoising, Normalization)
         ↓
output/denoised.wav
         ↓
scripts/02_transcribe.py (Local Whisper ASR with Timestamps)
         ↓
output/transcript.txt
```

---

## Commands

### Install

Create virtual environment:
```bash
python -m venv .venv
```

Windows:
```powershell
.venv\Scripts\activate
```

Then install dependencies:
```bash
pip install -r requirements.txt
```

### Denoise

```bash
python scripts/01_denoise.py
```

Expected output:
```text
output/denoised.wav
```

### Transcribe

```bash
python scripts/02_transcribe.py
```

Expected output:
```text
output/transcript.txt
```

---

## Optional Single Command Pipeline

To execute both stages sequentially:

```bash
python run_pipeline.py
```

---

## Configuration

In `scripts/02_transcribe.py`, model settings can be adjusted:

```python
MODEL_SIZE = "small"   # "tiny", "base", "small", "medium", "large-v3"
DEVICE = "auto"        # "auto", "cuda", "cpu"
COMPUTE_TYPE = "auto"  # "auto", "float16", "int8", "float32"
```

---

## Project Structure

```text
audio/
├── input/
│   └── war-intercept.wav
├── output/
│   ├── denoised.wav
│   └── transcript.txt
├── scripts/
│   ├── 01_denoise.py
│   └── 02_transcribe.py
├── run_pipeline.py
├── requirements.txt
├── README.md
└── .gitignore
```
