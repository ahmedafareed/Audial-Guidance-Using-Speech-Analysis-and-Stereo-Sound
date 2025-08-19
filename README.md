# Notebook -> Python package

This directory contains a converted set of Python modules extracted from the
original Jupyter notebook (`Full_implementation (1) (1).ipynb`).

Files of interest:

- `src/tts.py` — text-to-speech helper (calls Piper if available, otherwise writes silence)
- `src/coords.py` — Cartesian <-> spherical coordinate helpers
- `src/hrtf.py` — functions to load CIPIC-style HRTF and apply HRIR convolution
- `main.py` — small runner that demonstrates the pipeline
- `requirements.txt` — Python dependencies

Usage:

1. Create a venv and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Adjust paths in `main.py` (HRTF dataset root) if you want real HRTF output.

3. Run the demo:

```powershell
python main.py
```
