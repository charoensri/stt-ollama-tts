import os
import sounddevice as sd
import soundfile as sf
import numpy as np
from dotenv import load_dotenv

load_dotenv()

sample_rate = int(os.getenv("SAMPLE_RATE", "16000"))
seconds = int(os.getenv("RECORD_SECONDS", "5"))
input_device = os.getenv("INPUT_DEVICE") or None
output_device = os.getenv("OUTPUT_DEVICE") or None

print(f"Recording {seconds}s at {sample_rate} Hz...")
audio = sd.rec(
    int(seconds * sample_rate),
    samplerate=sample_rate,
    channels=1,
    dtype="float32",
    device=input_device,
)
sd.wait()

wav_path = "tmp_test_recording.wav"
sf.write(wav_path, audio, sample_rate)
print(f"Saved {wav_path}")

print("Playing back...")
data, sr = sf.read(wav_path, dtype="float32")
sd.play(data, sr, device=output_device)
sd.wait()
print("Done.")
