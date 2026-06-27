# Ollama Local Voice Assistant for WSL / Windows

A **no API key** local voice assistant pipeline:

```text
Microphone → Whisper/faster-whisper STT → Ollama LLM → Piper TTS → Speaker
```

This repo is tailored for **Windows + WSL2 / WSLg** setups where you already run Ollama locally.

---

## What requires API keys?

Nothing in the core stack.

- STT: `faster-whisper` runs locally
- LLM: `Ollama` runs locally
- TTS: `Piper` runs locally

You only need internet once to download Python packages, Ollama models, Whisper model weights, and Piper voice files.

---

## Folder structure

```text
ollama_voice_wsl_repo/
├─ app.py
├─ test_audio.py
├─ diagnose_wsl_audio.py
├─ requirements.txt
├─ .env.example
├─ setup_wsl.sh
├─ download_piper_voice.sh
└─ README.md
```

---

## 1. Prerequisites

### In Windows or WSL: Ollama running

If Ollama is installed in WSL:

```bash
ollama serve
ollama pull llama3.2:3b
```

If Ollama is installed on Windows, make sure WSL can reach it.

Try from WSL:

```bash
curl http://localhost:11434/api/tags
```

If that fails, try:

```bash
export OLLAMA_BASE_URL="http://$(grep nameserver /etc/resolv.conf | awk '{print $2}'):11434"
curl "$OLLAMA_BASE_URL/api/tags"
```

If Windows Ollama still refuses connections, start Ollama on Windows with a host binding that WSL can access, for example via an environment variable:

```powershell
setx OLLAMA_HOST 0.0.0.0:11434
```

Then restart Ollama.

---

## 2. Setup in WSL

```bash
cd ollama_voice_wsl_repo
chmod +x setup_wsl.sh download_piper_voice.sh
./setup_wsl.sh
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Download a Piper voice:

```bash
./download_piper_voice.sh
```

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` if needed:

```bash
nano .env
```

---

## 3. Test audio devices

List devices:

```bash
python diagnose_wsl_audio.py
```

Record and play back a short test:

```bash
python test_audio.py
```

If microphone does not work in WSL, run the app in Windows Python instead, or use a browser/WebRTC front-end later. WSLg audio output is usually easier than WSL microphone input.

---

## 4. Run the assistant

```bash
python app.py
```

Default behaviour:

- Press Enter
- Speak for a fixed number of seconds
- It transcribes your speech
- Sends text to Ollama
- Synthesizes response with Piper
- Plays the voice response

Quit with:

```text
q
```

---

## Useful `.env` settings

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b
WHISPER_MODEL=base.en
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
RECORD_SECONDS=5
SAMPLE_RATE=16000
PIPER_MODEL=models/en_US-lessac-medium.onnx
SYSTEM_PROMPT=You are a concise local voice assistant. Keep responses short and conversational.
```

For your RTX 1000 Ada 6GB VRAM, start with CPU STT first:

```env
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

Then try CUDA later:

```env
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

---

## Troubleshooting

### Ollama port already in use

That means Ollama is already running. Do not start another one. Test it:

```bash
curl http://localhost:11434/api/tags
```

### No microphone in WSL

Check:

```bash
python diagnose_wsl_audio.py
```

If no input device appears, WSL cannot see your Windows microphone. Options:

1. Run this repo with Windows Python instead of WSL Python.
2. Use a web UI / WebRTC frontend.
3. Record audio in Windows and pass WAV files into WSL.

### Piper command not found

Activate the venv:

```bash
source .venv/bin/activate
```

Then check:

```bash
which piper
piper --help
```

### TTS model missing

Run:

```bash
./download_piper_voice.sh
```

---

## Why this design?

This is intentionally modular:

- Replace `faster-whisper` with `whisper.cpp` later if you want lower-level control.
- Replace Piper with Coqui/Bark later if you want higher quality.
- Keep Ollama as the local reasoning layer.
- Add MCP/tool-calling inside `ask_ollama()` later.


```
//========= More robust example =========
python app_production.py
```

```
//------- .env ---------
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:latest
OLLAMA_TIMEOUT=120
OLLAMA_TEMPERATURE=0.4
OLLAMA_NUM_CTX=4096
WHISPER_MODEL=base.en
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
RECORD_SECONDS=5
SAMPLE_RATE=16000
INPUT_DEVICE=
OUTPUT_DEVICE=
PIPER_MODEL=en_US-lessac-medium.onnx
PIPER_EXE=piper
ASSISTANT_NAME=Luna
USER_LOCATION=Sydney, NSW, Australia
MAX_MEMORY_TURNS=8
SAVE_AUDIO=false
#Important change:
PIPER_MODEL=models/en_US-lessac-medium.onnx
#because your model is beside app.py, not inside a models folder.

```
