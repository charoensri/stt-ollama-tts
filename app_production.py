import os
import sys
import json
import time
import wave
import shutil
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime

import requests
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from faster_whisper import WhisperModel

# ============================================================
# Local Ollama Voice Assistant - Production-ready app.py
# Mic -> faster-whisper -> Ollama /api/chat -> Piper -> Speaker
# Features:
# - conversation memory
# - structured JSONL logging
# - safer error handling
# - model/path validation
# - reset/history/devices helper commands
# - Windows-friendly Piper handling
# ============================================================

load_dotenv()

APP_NAME = "Local Ollama Voice Assistant"
APP_VERSION = "2.0.0"

BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "tmp"
LOG_DIR = BASE_DIR / "logs"
TMP_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# -----------------------------
# Environment configuration
# -----------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

RECORD_SECONDS = int(os.getenv("RECORD_SECONDS", "5"))
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
INPUT_DEVICE = os.getenv("INPUT_DEVICE") or None
OUTPUT_DEVICE = os.getenv("OUTPUT_DEVICE") or None

# Your current folder has en_US-lessac-medium.onnx directly beside app.py, so default to that.
PIPER_MODEL = os.getenv("PIPER_MODEL", "en_US-lessac-medium.onnx")
PIPER_EXE = os.getenv("PIPER_EXE", "piper")

MAX_MEMORY_TURNS = int(os.getenv("MAX_MEMORY_TURNS", "8"))  # user+assistant pairs retained
SAVE_AUDIO = os.getenv("SAVE_AUDIO", "false").lower() == "true"

ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Luna")
USER_LOCATION = os.getenv("USER_LOCATION", "Sydney, NSW, Australia")

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    (
        f"You are {ASSISTANT_NAME}, a concise, friendly local voice assistant. "
        "You are running fully locally with no cloud API keys. "
        "Keep spoken answers short, natural, and useful. "
        "You DO have memory within this running session because the app sends prior turns. "
        "If the user asks about prior turns in this same session, use the conversation history. "
        f"The user's location is {USER_LOCATION}. "
        "If asked about the current date or time, use the current date/time supplied in the developer context message."
    ),
)

SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"conversation_{SESSION_ID}.jsonl"

# Conversation memory for Ollama /api/chat
chat_history = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
]

# -----------------------------
# Structured logging
# -----------------------------
def log_event(event_type: str, **fields):
    record = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "session_id": SESSION_ID,
        "event": event_type,
        **fields,
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_event(event_type: str, **fields):
    # Human-friendly console output + file JSONL logging.
    log_event(event_type, **fields)
    if event_type == "error":
        print(f"ERROR: {fields.get('message', 'Unknown error')}")
    elif event_type == "user_text":
        print(f"\nYou: {fields.get('text', '')}")
    elif event_type == "assistant_text":
        print(f"Assistant: {fields.get('text', '')}\n")


# -----------------------------
# Validation helpers
# -----------------------------
def resolve_piper_model() -> Path:
    """Resolve Piper model path robustly for Windows/Linux."""
    p = Path(PIPER_MODEL)
    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            BASE_DIR / p,
            BASE_DIR / "models" / p.name,
            Path.cwd() / p,
            Path.cwd() / "models" / p.name,
        ])

    for c in candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        "Piper voice model not found. Tried:\n" +
        "\n".join(f"  - {c}" for c in candidates) +
        "\n\nFix options:\n"
        "  1) Set PIPER_MODEL=en_US-lessac-medium.onnx in .env if the model is beside app.py\n"
        "  2) Or move en_US-lessac-medium.onnx and .json into a models/ folder\n"
    )


def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
        data = r.json()
        models = [m.get("name") for m in data.get("models", [])]
        log_event("ollama_check", base_url=OLLAMA_BASE_URL, models=models)
        if OLLAMA_MODEL not in models:
            print(f"WARNING: OLLAMA_MODEL={OLLAMA_MODEL!r} was not found exactly in ollama list.")
            print("Available models:")
            for m in models:
                print(f"  - {m}")
            print("The app will still try to use it; update .env if needed.\n")
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Start Ollama and verify: curl http://localhost:11434/api/tags. "
            f"Original error: {e}"
        )


def list_audio_devices():
    print("\nPortAudio devices visible from Python:\n")
    print(sd.query_devices())
    print("\nDefault input/output device:")
    print(sd.default.device)
    print()


# -----------------------------
# Audio capture / playback
# -----------------------------
def record_audio() -> Path:
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

    if SAVE_AUDIO:
        archived = LOG_DIR / f"input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        shutil.copyfile(wav_path, archived)
        log_event("audio_saved", path=str(archived))

    return wav_path


def play_wav(wav_path: Path):
    data, sr = sf.read(str(wav_path), dtype="float32")
    print("▶️  Playing response...")
    sd.play(data, sr, device=OUTPUT_DEVICE)
    sd.wait()


# -----------------------------
# STT / LLM / TTS
# -----------------------------
def load_stt_model() -> WhisperModel:
    print(f"Loading Whisper model: {WHISPER_MODEL} on {WHISPER_DEVICE}/{WHISPER_COMPUTE_TYPE}")
    model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )
    log_event("stt_model_loaded", model=WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    return model


def transcribe(stt_model: WhisperModel, wav_path: Path) -> str:
    print("🧠 Transcribing...")
    segments, info = stt_model.transcribe(
        str(wav_path),
        beam_size=5,
        vad_filter=True,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    log_event("transcription", text=text, language=getattr(info, "language", None))
    return text


def current_context_message() -> dict:
    # Supplied every turn so the local model can answer date/time questions.
    now = datetime.now().astimezone()
    return {
        "role": "system",
        "content": (
            "Current runtime context: "
            f"local_datetime={now.isoformat(timespec='seconds')}; "
            f"user_location={USER_LOCATION}; "
            "Use this context for current date/time/location questions."
        ),
    }


def trim_history():
    """Preserve system prompt; keep only the recent user/assistant turns."""
    global chat_history
    system_messages = [m for m in chat_history if m["role"] == "system"][:1]
    non_system = [m for m in chat_history if m["role"] != "system"]
    keep = MAX_MEMORY_TURNS * 2
    chat_history = system_messages + non_system[-keep:]


def ask_ollama(user_text: str) -> str:
    print(f"🤖 Asking Ollama model: {OLLAMA_MODEL}")

    chat_history.append({"role": "user", "content": user_text})
    trim_history()

    # Send current time as transient system message, but don't store it permanently.
    messages_for_request = [chat_history[0], current_context_message()] + chat_history[1:]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages_for_request,
        "stream": False,
        "options": {
            "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.4")),
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096")),
        },
    }

    r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=OLLAMA_TIMEOUT)
    if r.status_code == 404:
        raise RuntimeError(
            "Ollama /api/chat returned 404. Your running Ollama server may not support chat API. "
            "If needed, switch to /api/generate path; but your latest test showed /api/chat works."
        )
    r.raise_for_status()

    result = r.json()
    try:
        assistant_reply = result["message"]["content"].strip()
    except KeyError as e:
        raise RuntimeError(f"Unexpected Ollama response shape. Missing {e}. Full response: {result}")

    chat_history.append({"role": "assistant", "content": assistant_reply})
    trim_history()
    log_event("ollama_response", model=OLLAMA_MODEL, response=assistant_reply)
    return assistant_reply


def speak_with_piper(text: str) -> Path:
    model_path = resolve_piper_model()
    out_wav = TMP_DIR / "response.wav"

    print("🔊 Synthesizing speech with Piper...")
    cmd = [
        PIPER_EXE,
        "--model",
        str(model_path),
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
        stderr = proc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"Piper failed with exit code {proc.returncode}: {stderr}")

    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise RuntimeError("Piper did not create a valid output WAV file.")

    log_event("tts_generated", model=str(model_path), output=str(out_wav), text=text)
    return out_wav


# -----------------------------
# Commands and main loop
# -----------------------------
def print_help():
    print("Commands:")
    print("  Enter     Record voice")
    print("  q         Quit")
    print("  reset     Clear conversation memory")
    print("  history   Show recent memory")
    print("  devices   List audio devices")
    print("  help      Show this help")
    print()


def show_history():
    print("\n--- Recent conversation memory ---")
    for m in chat_history:
        role = m["role"]
        content = m["content"]
        if role == "system":
            content = content[:140] + "..." if len(content) > 140 else content
        print(f"[{role}] {content}")
    print("--- end ---\n")


def reset_memory():
    global chat_history
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    log_event("memory_reset")
    print("Conversation memory reset.\n")


def main():
    print(APP_NAME)
    print(f"Version: {APP_VERSION}")
    print("No API keys. Mic → Whisper → Ollama → Piper → Speaker")
    print(f"Session log: {LOG_FILE}")
    print_help()

    log_event("app_start", version=APP_VERSION)

    try:
        check_ollama()
        resolve_piper_model()
        stt_model = load_stt_model()
    except Exception as e:
        print_event("error", message=str(e))
        sys.exit(1)

    while True:
        try:
            cmd = input("Press Enter to record, or command (help/q/reset/history/devices): ").strip().lower()

            if cmd in {"q", "quit", "exit"}:
                print("Bye.")
                log_event("app_exit")
                break
            if cmd == "help":
                print_help()
                continue
            if cmd == "reset":
                reset_memory()
                continue
            if cmd == "history":
                show_history()
                continue
            if cmd == "devices":
                list_audio_devices()
                continue

            wav_path = record_audio()
            user_text = transcribe(stt_model, wav_path)

            if not user_text:
                print("I did not hear anything clear. Try again.\n")
                log_event("empty_transcription")
                continue

            print_event("user_text", text=user_text)

            assistant_reply = ask_ollama(user_text)
            print_event("assistant_text", text=assistant_reply)

            response_wav = speak_with_piper(assistant_reply)
            play_wav(response_wav)

        except KeyboardInterrupt:
            print("\nInterrupted. Bye.")
            log_event("keyboard_interrupt")
            break
        except requests.exceptions.RequestException as e:
            print_event("error", message=f"Network/API error talking to Ollama: {e}")
            print("Tip: verify Ollama with: curl http://localhost:11434/api/tags\n")
        except Exception as e:
            print_event("error", message=str(e))
            print("Tip: use 'devices' for audio issues, 'history' for memory, or 'reset' to clear memory.\n")


if __name__ == "__main__":
    main()
