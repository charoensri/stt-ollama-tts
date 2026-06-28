import os
import re
import json
import time
import wave
import queue
import tempfile
import threading
import subprocess
from dataclasses import dataclass, field

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import torch

from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator


# ============================================================
# CONFIG
# ============================================================

@dataclass
class AppConfig:
    # Audio
    sample_rate: int = 16000
    channels: int = 1

    # Silero VAD expects 512 samples at 16 kHz for streaming use
    vad_window_samples: int = 512

    # VAD behaviour
    vad_threshold: float = 0.50
    min_silence_duration_ms: int = 600
    speech_pad_ms: int = 120

    # Ignore tiny accidental noises
    min_utterance_ms: int = 500
    max_utterance_seconds: int = 20

    # Whisper
    whisper_model: str = "base"
    whisper_compute_type: str = "int8"
    whisper_device: str = "cpu"

    # Ollama
    ollama_url: str = "http://localhost:11434/api/chat"
    ollama_model: str = "llama3.2"

    # Piper
    piper_exe: str = "piper"
    piper_model: str = "en_US-lessac-medium.onnx"

    # Conversation
    system_prompt: str = (
        "You are a concise helpful local voice assistant. "
        "Keep responses short, natural, and spoken-friendly."
    )

    # Realtime TTS chunking
    tts_min_chunk_chars: int = 80
    tts_max_chunk_chars: int = 220

    # If True, user voice can interrupt TTS.
    # Best with headphones to avoid the mic hearing the speaker.
    allow_barge_in: bool = True


# ============================================================
# REALTIME VOICE ASSISTANT
# ============================================================

class RealtimeVoiceAssistant:
    def __init__(self, config: AppConfig):
        self.cfg = config

        self.audio_q = queue.Queue()
        self.tts_q = queue.Queue()

        self.running = True
        self.is_speaking = False
        self.interrupt_event = threading.Event()

        self.messages = [
            {"role": "system", "content": self.cfg.system_prompt}
        ]

        print("Loading Silero VAD...")
        torch.set_num_threads(1)

        # onnx=True uses ONNX runtime backend where available.
        # If you hit issues, change to: load_silero_vad()
        self.vad_model = load_silero_vad(onnx=True)

        self.vad_iterator = VADIterator(
            self.vad_model,
            sampling_rate=self.cfg.sample_rate,
            threshold=self.cfg.vad_threshold,
            min_silence_duration_ms=self.cfg.min_silence_duration_ms,
            speech_pad_ms=self.cfg.speech_pad_ms,
        )

        print("Loading Whisper model...")
        self.whisper = WhisperModel(
            self.cfg.whisper_model,
            device=self.cfg.whisper_device,
            compute_type=self.cfg.whisper_compute_type,
        )

        self.tts_thread = threading.Thread(
            target=self.tts_worker,
            daemon=True
        )
        self.tts_thread.start()

    # ------------------------------------------------------------
    # AUDIO INPUT
    # ------------------------------------------------------------

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"\nAudio status: {status}")

        self.audio_q.put(bytes(indata))

    def bytes_to_float_tensor(self, audio_bytes: bytes) -> torch.Tensor:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio_np = audio_np / 32768.0
        return torch.from_numpy(audio_np)

    def save_wav_bytes(self, pcm_bytes: bytes, path: str):
        with wave.open(path, "wb") as wf:
            wf.setnchannels(self.cfg.channels)
            wf.setsampwidth(2)  # int16
            wf.setframerate(self.cfg.sample_rate)
            wf.writeframes(pcm_bytes)

    # ------------------------------------------------------------
    # TTS INTERRUPTION
    # ------------------------------------------------------------

    def clear_tts_queue(self):
        try:
            while True:
                self.tts_q.get_nowait()
                self.tts_q.task_done()
        except queue.Empty:
            pass

    def interrupt_tts(self):
        if self.is_speaking and self.cfg.allow_barge_in:
            print("\n[interrupting speech]")
            self.interrupt_event.set()
            self.clear_tts_queue()
            sd.stop()

    # ------------------------------------------------------------
    # STT
    # ------------------------------------------------------------

    def transcribe_pcm_bytes(self, pcm_bytes: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name

        try:
            self.save_wav_bytes(pcm_bytes, wav_path)

            segments, info = self.whisper.transcribe(
                wav_path,
                beam_size=1,
                vad_filter=False,
                language="en",
            )

            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text

        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    # ------------------------------------------------------------
    # OLLAMA STREAMING
    # ------------------------------------------------------------

    def ollama_stream(self, user_text: str):
        self.messages.append({"role": "user", "content": user_text})

        payload = {
            "model": self.cfg.ollama_model,
            "messages": self.messages,
            "stream": True,
            "options": {
                "temperature": 0.4
            }
        }

        full_response = ""

        try:
            with requests.post(
                self.cfg.ollama_url,
                json=payload,
                stream=True,
                timeout=120,
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    data = json.loads(line.decode("utf-8"))

                    if "message" in data and "content" in data["message"]:
                        token = data["message"]["content"]
                        full_response += token
                        yield token

                    if data.get("done") is True:
                        break

        except Exception as e:
            error_text = f"Sorry, I could not reach Ollama. Error: {e}"
            print(error_text)
            full_response = error_text
            yield error_text

        finally:
            if full_response.strip():
                self.messages.append(
                    {"role": "assistant", "content": full_response.strip()}
                )

            # Keep memory bounded
            if len(self.messages) > 16:
                self.messages = [self.messages[0]] + self.messages[-14:]

    # ------------------------------------------------------------
    # STREAMED TTS CHUNKING
    # ------------------------------------------------------------

    def should_flush_tts_chunk(self, buffer: str) -> bool:
        text = buffer.strip()

        if len(text) >= self.cfg.tts_max_chunk_chars:
            return True

        if len(text) >= self.cfg.tts_min_chunk_chars:
            if re.search(r"[.!?]\s*$", text):
                return True

        return False

    def stream_answer_to_tts(self, user_text: str):
        print("AI: ", end="", flush=True)

        buffer = ""

        for token in self.ollama_stream(user_text):
            print(token, end="", flush=True)
            buffer += token

            if self.should_flush_tts_chunk(buffer):
                self.tts_q.put(buffer.strip())
                buffer = ""

        if buffer.strip():
            self.tts_q.put(buffer.strip())

        print()

    # ------------------------------------------------------------
    # PIPER TTS
    # ------------------------------------------------------------

    def tts_worker(self):
        while self.running:
            try:
                text = self.tts_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if not text.strip():
                self.tts_q.task_done()
                continue

            if self.interrupt_event.is_set():
                self.tts_q.task_done()
                continue

            try:
                self.speak_with_piper(text)
            except Exception as e:
                print(f"\nTTS error: {e}")
            finally:
                self.tts_q.task_done()

    def speak_with_piper(self, text: str):
        self.is_speaking = True

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name

        try:
            # Generate audio with Piper
            cmd = [
                self.cfg.piper_exe,
                "--model",
                self.cfg.piper_model,
                "--output_file",
                wav_path,
            ]

            subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )

            if self.interrupt_event.is_set():
                return

            data, sr = sf.read(wav_path, dtype="float32")

            sd.play(data, sr)

            # sd.wait() returns when playback finishes or sd.stop() is called.
            sd.wait()

        finally:
            self.is_speaking = False
            self.interrupt_event.clear()

            try:
                os.remove(wav_path)
            except OSError:
                pass

    # ------------------------------------------------------------
    # MAIN REALTIME LOOP
    # ------------------------------------------------------------

    def run(self):
        print()
        print("===================================================")
        print(" Local Realtime Voice Assistant v2")
        print(" Mic → Silero VAD → Whisper → Ollama → Piper")
        print(" Say something. Press Ctrl+C to quit.")
        print("===================================================")
        print()

        speech_buffer = bytearray()
        in_speech = False
        speech_started_at = None

        blocksize = self.cfg.vad_window_samples

        try:
            with sd.RawInputStream(
                samplerate=self.cfg.sample_rate,
                blocksize=blocksize,
                dtype="int16",
                channels=self.cfg.channels,
                callback=self.audio_callback,
            ):
                while self.running:
                    chunk = self.audio_q.get()

                    # Make sure chunk is exactly 512 samples / 1024 bytes.
                    expected_bytes = self.cfg.vad_window_samples * 2
                    if len(chunk) != expected_bytes:
                        continue

                    audio_tensor = self.bytes_to_float_tensor(chunk)

                    speech_event = self.vad_iterator(
                        audio_tensor,
                        return_seconds=False
                    )

                    if speech_event:
                        if "start" in speech_event:
                            in_speech = True
                            speech_started_at = time.time()
                            speech_buffer = bytearray()

                            if self.cfg.allow_barge_in:
                                self.interrupt_tts()

                            print("\n[start speech]")

                        if in_speech:
                            speech_buffer.extend(chunk)

                        if "end" in speech_event:
                            print("[end speech]")

                            in_speech = False

                            duration_ms = (
                                len(speech_buffer)
                                / 2
                                / self.cfg.sample_rate
                                * 1000
                            )

                            if duration_ms < self.cfg.min_utterance_ms:
                                print("[ignored short noise]")
                                speech_buffer = bytearray()
                                self.vad_iterator.reset_states()
                                continue

                            pcm = bytes(speech_buffer)
                            speech_buffer = bytearray()

                            print("Transcribing...")
                            user_text = self.transcribe_pcm_bytes(pcm)

                            if not user_text:
                                print("[empty transcription]")
                                self.vad_iterator.reset_states()
                                continue

                            print(f"You: {user_text}")

                            if user_text.lower().strip() in {
                                "quit",
                                "exit",
                                "stop",
                                "goodbye",
                            }:
                                print("Exiting.")
                                self.running = False
                                break

                            self.stream_answer_to_tts(user_text)

                            self.vad_iterator.reset_states()

                    else:
                        if in_speech:
                            speech_buffer.extend(chunk)

                            if speech_started_at:
                                elapsed = time.time() - speech_started_at
                                if elapsed > self.cfg.max_utterance_seconds:
                                    print("[max utterance reached]")
                                    in_speech = False

                                    pcm = bytes(speech_buffer)
                                    speech_buffer = bytearray()

                                    user_text = self.transcribe_pcm_bytes(pcm)
                                    if user_text:
                                        print(f"You: {user_text}")
                                        self.stream_answer_to_tts(user_text)

                                    self.vad_iterator.reset_states()

        except KeyboardInterrupt:
            print("\nCtrl+C received. Exiting.")
        finally:
            self.running = False
            sd.stop()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    config = AppConfig(
        # Change if needed
        ollama_model="llama3.2",

        # This must match your Piper model file in current folder or full path.
        piper_model="models\en_US-lessac-medium.onnx",

        # Keep this True if piper.exe is in PATH.
        # Otherwise use full path, for example:
        # piper_exe=r"C:\Users\chars1\python311env\ollama_voice_wsl_repo\piper\piper.exe"
        piper_exe="piper",

        # For better accuracy but slower:
        # whisper_model="small",
        whisper_model="base",

        allow_barge_in=True,
    )

    app = RealtimeVoiceAssistant(config)
    app.run()