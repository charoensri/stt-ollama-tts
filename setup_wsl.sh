#!/usr/bin/env bash
set -e

sudo apt-get update
sudo apt-get install -y \
  python3 python3-venv python3-pip \
  ffmpeg curl wget \
  portaudio19-dev libasound2-dev libsndfile1 \
  pulseaudio-utils alsa-utils

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt

mkdir -p models tmp

echo ""
echo "Setup complete. Next:"
echo "  source .venv/bin/activate"
echo "  ./download_piper_voice.sh"
echo "  cp .env.example .env"
echo "  python diagnose_wsl_audio.py"
echo "  python test_audio.py"
echo "  python app.py"
