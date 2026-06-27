import os
import sys
import json
import time
import queue
import shutil
import threading
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from faster_whisper import WhisperModel

# ============================================================
# Local Ollama Voice Assistant - Semi-Realtime app.py
# Mic chunks -> silence detection -> Whisper -> Ollama streaming -> Piper sentence TTS -> Speaker
# Minimal upgrade from the working app:
# - no fixed 5-second recording requirement
# - records until speech pause/silence
# - streams Ollama tokens
# - speaks sentence-by-sentence while the LLM is still generating
# - keeps conversation memory
# - logs JSONL events
# ============================================================

load_dotenv()

APP_NAME = "Local Ollama Voice Assistant - Semi-Realtime"
APP_VERSION = "2.1.0"

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

SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
INPUT_DEVICE = os.getenv("INPUT_DEVICE") or None
OUTPUT_DEVICE = os.getenv("OUTPUT_DEVICE") or None

PIPER_MODEL = os.getenv("PIPER_MODEL", "en_US-lessac-medium.onnx")
PIPER_EXE = os.getenv("PIPER_EXE", "piper")

MAX_MEMORY_TURNS = int(os.getenv("MAX_MEMORY_TURNS", "8"))
SAVE_AUDIO = os.getenv("SAVE_AUDIO", "false").lower() == "true"

ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Luna")
USER_LOCATION = os.getenv("USER_LOCATION", "Sydney, NSW, Australia")

# Semi-realtime recording knobs
CHUNK_SECONDS = float(os.getenv("REALTIME_CHUNK_SECONDS", "0.25"))
MAX_RECORD_SECONDS = float(os.getenv("REALTIME_MAX_RECORD_SECONDS", "20"))
MIN_SPEECH_SECONDS = float(os.getenv("REALTIME_MIN_SPEECH_SECONDS", "0.4"))
SILENCE_SECONDS = float(os.getenv("REALTIME_SILENCE_SECONDS", "1.0"))
RMS_THRESHOLD = float(os.getenv("REALTIME_RMS_THRESHOLD", "0.010"))
PRE_ROLL_SECONDS = float(os.getenv("REALTIME_PRE_ROLL_SECONDS", "0.5"))

SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    (
        f"You are {ASSISTANT_NAME}, a concise, friendly local voice assistant. "
        "You are running fully locally with no cloud API keys. "
        "Keep spoken answers short, natural, and useful. "
        "You DO have memory within this running session because the app sends prior turns. "
        "If the user asks about prior turns in this same session, use the conversation history. "
        f"The user's location is {USER_LOCATION}. "
        "If asked about current date or time, use the current date/time supplied in the runtime context message."
    ),
)

SESSION_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"conversation_semirealtime_{SESSION_ID}.jsonl"

chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# -----------------------------
# Logging
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


def print_error(message: str):
    log_event("error", message=message)
    print(f"ERROR: {message}")

# -----------------------------
# Validation helpers
# -----------------------------
def resolve_piper_model() -> Path:
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
        "\nSet PIPER_MODEL=en_US-lessac-medium.onnx if the model is beside app.py."
    )


def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
        r.raise_for_status()
        data = r.json()
        models = [m.get("name") for m in data.get("models", [])]
        log_event("ollama_check", base_url=OLLAMA_BASE_URL, models=models)
        if OLLAMA_MODEL not in models:
            print(f"WARNING: OLLAMA_MODEL={OLLAMA_MODEL!r} not found exactly in Ollama model list.")
            print("Available models:")
            for m in models:
                print(f"  - {m}")
            print()
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. Verify with: curl http://localhost:11434/api/tags. Original error: {e}"
        )


def list_audio_devices():
    print("\nPortAudio devices visible from Python:\n")
    print(sd.query_devices())
    print("\nDefault input/output device:")
    print(sd.default.device)
    print()

# -----------------------------
# Semi-realtime recording until silence
# -----------------------------
def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float32)))))


def record_until_silence() -> Path | None:
    """Record small chunks until the user stops speaking."""
    chunk_samples = int(SAMPLE_RATE * CHUNK_SECONDS)
    max_chunks = int(MAX_RECORD_SECONDS / CHUNK_SECONDS)
    silence_chunks_needed = max(1, int(SILENCE_SECONDS / CHUNK_SECONDS))
    min_speech_chunks = max(1, int(MIN_SPEECH_SECONDS / CHUNK_SECONDS))
    pre_roll_chunks = max(0, int(PRE_ROLL_SECONDS / CHUNK_SECONDS))

    print("🎙️  Listening... speak now. I will stop when you pause.")

    frames = []
    pre_roll = []
    speech_started = False
    speech_chunks = 0
    silence_chunks = 0

    for i in range(max_chunks):
        chunk = sd.rec(
            chunk_samples,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=INPUT_DEVICE,
        )
        sd.wait()
        level = rms(chunk)
        is_speech = level >= RMS_THRESHOLD

        if not speech_started:
            pre_roll.append(chunk)
            if len(pre_roll) > pre_roll_chunks:
                pre_roll.pop(0)

            if is_speech:
                speech_started = True
                frames.extend(pre_roll)
                frames.append(chunk)
                speech_chunks = 1
                silence_chunks = 0
                print("🟢 Speech detected...")
            else:
                if i % max(1, int(1 / CHUNK_SECONDS)) == 0:
                    print(f"   waiting... rms={level:.4f}")
            continue

        frames.append(chunk)

        if is_speech:
            speech_chunks += 1
            silence_chunks = 0
        else:
            silence_chunks += 1

        if speech_chunks >= min_speech_chunks and silence_chunks >= silence_chunks_needed:
            break

    if not frames or not speech_started:
        print("I did not detect speech. Try again.\n")
        log_event("no_speech_detected")
        return None

    audio = np.concatenate(frames, axis=0)
    wav_path = TMP_DIR / "input_realtime.wav"
    sf.write(wav_path, audio, SAMPLE_RATE)

    duration = len(audio) / SAMPLE_RATE
    log_event("audio_recorded_until_silence", path=str(wav_path), duration_seconds=round(duration, 2))

    if SAVE_AUDIO:
        archived = LOG_DIR / f"input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        shutil.copyfile(wav_path, archived)
        log_event("audio_saved", path=str(archived))

    print(f"🔴 Captured {duration:.1f}s audio.\n")
    return wav_path

# -----------------------------
# STT
# -----------------------------
def load_stt_model() -> WhisperModel:
    print(f"Loading Whisper model: {WHISPER_MODEL} on {WHISPER_DEVICE}/{WHISPER_COMPUTE_TYPE}")
    model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    log_event("stt_model_loaded", model=WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)
    return model


def transcribe(stt_model: WhisperModel, wav_path: Path) -> str:
    print("🧠 Transcribing...")
    segments, info = stt_model.transcribe(str(wav_path), beam_size=5, vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    log_event("transcription", text=text, language=getattr(info, "language", None))
    return text

# -----------------------------
# Memory + Ollama streaming
# -----------------------------
def current_context_message() -> dict:
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
    global chat_history
    system_messages = [m for m in chat_history if m["role"] == "system"][:1]
    non_system = [m for m in chat_history if m["role"] != "system"]
    chat_history = system_messages + non_system[-(MAX_MEMORY_TURNS * 2):]


def sentence_split(buffer: str):
    """Return complete spoken chunks and remainder."""
    end_marks = ".!?\n"
    chunks = []
    start = 0
    for idx, ch in enumerate(buffer):
        if ch in end_marks:
            piece = buffer[start:idx + 1].strip()
            if piece:
                chunks.append(piece)
            start = idx + 1
    remainder = buffer[start:].strip()
    return chunks, remainder


def ask_ollama_stream(user_text: str, tts_queue: queue.Queue) -> str:
    print(f"🤖 Asking Ollama model: {OLLAMA_MODEL} (streaming)")

    chat_history.append({"role": "user", "content": user_text})
    trim_history()

    messages_for_request = [chat_history[0], current_context_message()] + chat_history[1:]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages_for_request,
        "stream": True,
        "options": {
            "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.4")),
            "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096")),
        },
    }

    full_text = ""
    pending_sentence_buffer = ""

    with requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
        stream=True,
    ) as r:
        r.raise_for_status()
        print("Assistant: ", end="", flush=True)

        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line:
                continue

            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            if "message" in event and "content" in event["message"]:
                token = event["message"]["content"]
                if token:
                    print(token, end="", flush=True)
                    full_text += token
                    pending_sentence_buffer += token

                    complete_sentences, pending_sentence_buffer = sentence_split(pending_sentence_buffer)
                    for sentence in complete_sentences:
                        # Sentence-level TTS starts before the full answer finishes.
                        tts_queue.put(sentence)

            if event.get("done") is True:
                break

    if pending_sentence_buffer.strip():
        tts_queue.put(pending_sentence_buffer.strip())

    print("\n")

    full_text = full_text.strip()
    chat_history.append({"role": "assistant", "content": full_text})
    trim_history()
    log_event("ollama_stream_response", model=OLLAMA_MODEL, response=full_text)
    return full_text

# -----------------------------
# Piper TTS worker
# -----------------------------
def synthesize_with_piper(text: str, index: int) -> Path:
    model_path = resolve_piper_model()
    out_wav = TMP_DIR / f"response_{index}.wav"

    cmd = [PIPER_EXE, "--model", str(model_path), "--output_file", str(out_wav)]
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
        raise RuntimeError("Piper did not create a valid WAV file.")

    log_event("tts_sentence_generated", text=text, output=str(out_wav))
    return out_wav


def play_wav(wav_path: Path):
    data, sr = sf.read(str(wav_path), dtype="float32")
    sd.play(data, sr, device=OUTPUT_DEVICE)
    sd.wait()


def tts_worker(tts_queue: queue.Queue, stop_token: object):
    index = 0
    while True:
        item = tts_queue.get()
        if item is stop_token:
            tts_queue.task_done()
            break

        text = str(item).strip()
        if not text:
            tts_queue.task_done()
            continue

        try:
            index += 1
            print(f"🔊 Speaking: {text}")
            wav_path = synthesize_with_piper(text, index)
            play_wav(wav_path)
        except Exception as e:
            print_error(f"TTS worker error: {e}")
        finally:
            tts_queue.task_done()

# -----------------------------
# Commands
# -----------------------------
def print_help():
    print("Commands:")
    print("  Enter     Listen until silence, then respond")
    print("  q         Quit")
    print("  reset     Clear conversation memory")
    print("  history   Show recent memory")
    print("  devices   List audio devices")
    print("  help      Show help")
    print()


def show_history():
    print("\n--- Recent conversation memory ---")
    for m in chat_history:
        role = m["role"]
        content = m["content"]
        if role == "system":
            content = content[:160] + "..." if len(content) > 160 else content
        print(f"[{role}] {content}")
    print("--- end ---\n")


def reset_memory():
    global chat_history
    chat_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    log_event("memory_reset")
    print("Conversation memory reset.\n")

# -----------------------------
# Main loop
# -----------------------------
def main():
    print(APP_NAME)
    print(f"Version: {APP_VERSION}")
    print("No API keys. Semi-realtime Mic → Whisper → streaming Ollama → sentence Piper → Speaker")
    print(f"Session log: {LOG_FILE}")
    print(f"Semi-realtime settings: chunk={CHUNK_SECONDS}s, silence={SILENCE_SECONDS}s, threshold={RMS_THRESHOLD}")
    print_help()

    log_event("app_start", version=APP_VERSION)

    try:
        check_ollama()
        resolve_piper_model()
        stt_model = load_stt_model()
    except Exception as e:
        print_error(str(e))
        sys.exit(1)

    while True:
        try:
            cmd = input("Press Enter to speak, or command (help/q/reset/history/devices): ").strip().lower()

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

            wav_path = record_until_silence()
            if wav_path is None:
                continue

            user_text = transcribe(stt_model, wav_path)
            if not user_text:
                print("I did not hear anything clear. Try again.\n")
                log_event("empty_transcription")
                continue

            print(f"\nYou: {user_text}")
            log_event("user_text", text=user_text)

            tts_queue = queue.Queue()
            stop_token = object()
            worker = threading.Thread(target=tts_worker, args=(tts_queue, stop_token), daemon=True)
            worker.start()

            assistant_text = ask_ollama_stream(user_text, tts_queue)
            log_event("assistant_text", text=assistant_text)

            # Wait for all generated speech to finish.
            tts_queue.put(stop_token)
            tts_queue.join()
            worker.join(timeout=2)

        except KeyboardInterrupt:
            print("\nInterrupted. Bye.")
            log_event("keyboard_interrupt")
            break
        except requests.exceptions.RequestException as e:
            print_error(f"Network/API error talking to Ollama: {e}")
            print("Tip: verify Ollama with: curl http://localhost:11434/api/tags\n")
        except Exception as e:
            print_error(str(e))
            print("Tip: use 'devices' for audio issues, 'history' for memory, or 'reset' to clear memory.\n")


if __name__ == "__main__":
    main()
