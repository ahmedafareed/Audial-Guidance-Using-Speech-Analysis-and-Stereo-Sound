"""Minimal visualization helpers (matplotlib optional)."""

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def plot_waveform(audio: np.ndarray, fs: int = 44100, title: str = "Waveform"):
    if plt is None:
        print("matplotlib not available; skipping plot")
        return
    t = np.arange(len(audio)) / fs
    plt.figure(figsize=(8, 2))
    plt.plot(t, audio)
    plt.xlabel("Time (s)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
