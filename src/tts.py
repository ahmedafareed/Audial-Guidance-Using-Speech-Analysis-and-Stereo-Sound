"""Text-to-speech utilities.

Provides a function `text_to_speech_piper` which attempts to call the `piper` CLI
if available, otherwise writes a short silent WAV as a fallback so the pipeline
can continue without hard failure. This keeps the module safe to run locally
and on CI where Piper may not be installed.
"""
import subprocess
import shutil
import numpy as np
import soundfile as sf


def text_to_speech_piper(text: str, output_filename: str, model_path: str = None) -> str:
    """Convert text to speech using the Piper CLI if available.

    If `piper` is available on PATH and `model_path` is given (or defaulted),
    it will invoke Piper to generate `output_filename`. If Piper is not
    available the function writes a short silent WAV file as a placeholder
    and returns the filename.

    Returns the path to the generated audio file.
    """
    print(f"[TTS] Converting text to speech for: '{text}' -> {output_filename}")

    # Prefer a provided model path, otherwise the caller is expected to
    # place the correct model if they want real TTS.
    if model_path is None:
        model_path = "/content/speech/ar-fareed-medium.onnx"  # informational default

    piper_path = shutil.which("piper")
    if piper_path:
        cmd = [piper_path, "--model", model_path, "--output_file", output_filename]
        # Provide text on stdin
        try:
            proc = subprocess.run(cmd, input=text.encode("utf-8"), check=True)
            print(f"[TTS] Piper produced: {output_filename}")
            return output_filename
        except subprocess.CalledProcessError as e:
            print(f"[TTS] Piper failed: {e}")
            # fallthrough to create placeholder
    else:
        print("[TTS] Piper binary not found on PATH. Creating placeholder audio.")

    # Fallback: write 0.5s of silence at 44.1kHz (mono)
    sr = 44100
    duration_s = 0.5
    silence = np.zeros(int(sr * duration_s), dtype=np.float32)
    sf.write(output_filename, silence, sr)
    print(f"[TTS] Wrote placeholder (silence) to: {output_filename}")
    return output_filename
