"""HRTF loading and application utilities.

These functions wrap loading a CIPIC-style MAT file and convolving an input
mono audio signal with the selected HRIRs to produce stereo output.
"""

import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import soundfile as sf
import librosa


def load_hrtf_data(hrtf_base_dir: str, subject_id: str = "003") -> dict:
    """Load HRTF data for the specified subject.

    hrtf_base_dir should be the directory that contains the unpacked
    `cipic-hrtf-database/standard_hrir_database` folder (or similar).

    Returns a dict with keys: 'hrir_l', 'hrir_r', 'azimuths', 'elevations'
    """
    hrtf_path = os.path.join(hrtf_base_dir, f"standard_hrir_database/subject_{subject_id}")
    print(f"[HRTF] Loading data from: {hrtf_path}")

    if not os.path.exists(hrtf_path):
        raise FileNotFoundError(f"HRTF data path '{hrtf_path}' does not exist. Please provide the correct path.")

    mat_contents = sio.loadmat(os.path.join(hrtf_path, "hrir_final.mat"))
    hrir_l = mat_contents.get("hrir_l")
    hrir_r = mat_contents.get("hrir_r")
    if hrir_l is None or hrir_r is None:
        raise KeyError("hrir_l or hrir_r not found in MAT file.")

    azimuths = np.array([
        -80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10,
        -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65, 80
    ], dtype=float)

    elevations = np.array([-45 + 5.625 * i for i in range(50)], dtype=float)

    print("[HRTF] Loaded HRIR arrays and defined angle grids")
    return {
        "hrir_l": hrir_l,
        "hrir_r": hrir_r,
        "azimuths": azimuths,
        "elevations": elevations,
    }


def apply_hrtf(audio_filename: str, azimuth: float, elevation: float, hrtf_data: dict) -> str:
    """Apply HRTF to the audio file based on the given azimuth and elevation.

    Returns the filename of the produced stereo file (with suffix `_hrtf.wav`).
    """
    if azimuth is None:
        print("[HRTF] Skipping HRTF application due to invalid azimuth.")
        return None

    print(f"[HRTF] Applying HRTF for azimuth {azimuth}°, elevation {elevation}° to '{audio_filename}'")

    audio_data, fs = sf.read(audio_filename)

    # If audio is stereo, convert to mono
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Resample if necessary
    fs_desired = 44100
    if fs != fs_desired:
        audio_data = librosa.resample(audio_data, orig_sr=fs, target_sr=fs_desired)
        fs = fs_desired

    hrir_l = hrtf_data["hrir_l"]
    hrir_r = hrtf_data["hrir_r"]
    azimuths = hrtf_data["azimuths"]
    elevations = hrtf_data["elevations"]

    # Clip azimuth / elevation to HRTF ranges
    if azimuth < -80 or azimuth > 80:
        azimuth = float(np.clip(azimuth, -80, 80))
    if elevation < -45 or elevation > 90:
        elevation = float(np.clip(elevation, -45, 90))

    azimuth_idx = np.abs(azimuths - azimuth).argmin()
    elevation_idx = np.abs(elevations - elevation).argmin()

    hrir_l_selected = hrir_l[azimuth_idx, elevation_idx, :]
    hrir_r_selected = hrir_r[azimuth_idx, elevation_idx, :]

    audio_left = signal.convolve(audio_data, hrir_l_selected, mode="full")
    audio_right = signal.convolve(audio_data, hrir_r_selected, mode="full")

    min_len = min(len(audio_left), len(audio_right))
    audio_left = audio_left[:min_len]
    audio_right = audio_right[:min_len]

    audio_stereo = np.vstack((audio_left, audio_right)).T

    max_val = np.max(np.abs(audio_stereo))
    if max_val > 0:
        audio_stereo = audio_stereo / max_val

    output_filename = audio_filename.replace('.wav', '_hrtf.wav')
    sf.write(output_filename, audio_stereo, fs)
    print(f"[HRTF] Saved HRTF-processed audio: {output_filename}")
    return output_filename
