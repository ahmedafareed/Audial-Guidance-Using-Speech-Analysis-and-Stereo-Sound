"""Small audio helper functions."""
import numpy as np
import soundfile as sf


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    maxv = np.max(np.abs(audio))
    if maxv == 0:
        return audio
    return audio / maxv


def read_mono(path: str, sr: int = None):
    data, fs = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr is not None and fs != sr:
        # Defer to librosa for resampling in calling code to avoid a hard dependency here
        raise RuntimeError("Resampling requested; caller should perform resampling with librosa")
    return data, fs
