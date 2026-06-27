import os
import json
import tempfile
import subprocess
from pathlib import Path

import requests
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a concise local voice assistant. Keep responses short and conversational.",
)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "5"))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
INPUT_DEVICE = os.getenv("INPUT_DEVICE") or None
OUTPUT_DEVICE = os.getenv("OUTPUT_DEVICE") or None

PIPER_MODEL = Path(os.getenv("PIPER_MODEL", "models/en_US-lessac-medium.onnx"))
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)


def record_audio() -> Path:
    """Record microphone audio to a temporary WAV file."""
    print(f"🎙️  Recording for {RECORD_SECONDS} seconds...")
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=INPUT_DEVICE,
    )
    sd.wait()

    wav_path = TMP_DIR / "input.wav"
    sf.write(wav_path, audio, SAMPLE_RATE)
    return wav_path


def load_stt_model() -> WhisperModel:
    print(f"Loading Whisper model: {WHISPER_MODEL} on {WHISPER_DEVICE}/{WHISPER_COMPUTE_TYPE}")
    return WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )


def transcribe(model: WhisperModel, wav_path: Path) -> str:
    print("🧠 Transcribing...")
    segments, info = model.transcribe(
        str(wav_path),
        beam_size=5,
        vad_filter=True,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text


def ask_ollama(user_text: str) -> str:
    print(f"🤖 Asking Ollama model: {OLLAMA_MODEL}")
    url = f"{OLLAMA_BASE_URL}/api/chat" #conversation state (if supported)
    #url = f"{OLLAMA_BASE_URL}/api/generate" #single-shot LLM call (stateless)
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"].strip()


def speak_with_piper(text: str) -> Path:
    if not PIPER_MODEL.exists():
        raise FileNotFoundError(
            f"Piper model not found: {PIPER_MODEL}. Run ./download_piper_voice.sh first."
        )

    out_wav = TMP_DIR / "response.wav"
    print("🔊 Synthesizing speech with Piper...")

    # Feed text via stdin to avoid shell quoting problems.
    cmd = [
        "piper",
        "--model",
        str(PIPER_MODEL),
        "--output_file",
        str(out_wav),
    ]
    proc = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))

    return out_wav


def play_wav(wav_path: Path):
    print("▶️  Playing response...")
    data, sr = sf.read(wav_path, dtype="float32")
    sd.play(data, sr, device=OUTPUT_DEVICE)
    sd.wait()


def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            f"Check Ollama is running and OLLAMA_BASE_URL is correct. Original error: {e}"
        )


def main():
    print("Local Ollama Voice Assistant")
    print("No API keys. Mic → Whisper → Ollama → Piper → Speaker")
    print("Type 'q' then Enter to quit. Press Enter to speak.\n")

    check_ollama()
    stt_model = load_stt_model()

    while True:
        cmd = input("Press Enter to record, or q to quit: ").strip().lower()
        if cmd in {"q", "quit", "exit"}:
            print("Bye.")
            break

        try:
            wav_path = record_audio()
            user_text = transcribe(stt_model, wav_path)

            if not user_text:
                print("I did not hear anything clear. Try again.")
                continue

            print(f"\nYou: {user_text}")
            answer = ask_ollama(user_text)
            print(f"Assistant: {answer}\n")

            response_wav = speak_with_piper(answer)
            play_wav(response_wav)

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            print("Tip: run python diagnose_wsl_audio.py and python test_audio.py if this is audio-related.\n")


if __name__ == "__main__":
    main()
