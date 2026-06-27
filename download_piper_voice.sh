#!/usr/bin/env bash
set -e
mkdir -p models
cd models

VOICE="en_US-lessac-medium"
BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"

if [ ! -f "${VOICE}.onnx" ]; then
  wget -O "${VOICE}.onnx" "${BASE}/${VOICE}.onnx"
fi

if [ ! -f "${VOICE}.onnx.json" ]; then
  wget -O "${VOICE}.onnx.json" "${BASE}/${VOICE}.onnx.json"
fi

echo "Downloaded Piper voice to models/${VOICE}.onnx"
