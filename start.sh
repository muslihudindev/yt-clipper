#!/bin/bash
. ~/.venvs/ai-clipper/bin/activate
python ai_clipper.py --file urls.txt --continue-on-error --clips 3 --whisper-model medium --language id --tts --gpu --tts-voice af_bella --face-track --gpu
