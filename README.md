# AI Clipper

YouTube-to-TikTok clip pipeline. Downloads a video, transcribes it, uses AI to find viral moments, and outputs upload-ready vertical clips with hook text, subtitles, TTS voiceover, face tracking, and GPU acceleration — all in one command.

## Quick Start

```bash
# 1. Install dependencies
pip install openai-whisper json-repair
pip install kokoro soundfile numpy  # optional: TTS
pip install opencv-python           # optional: face tracking

# 2. Setup AI backend
export OPENROUTER_API_KEY="sk-or-..."  # free at https://openrouter.ai/keys

# 3. Create urls.txt
cp urls.txt.example urls.txt
# Add your YouTube URLs

# 4. Run
python ai_clipper.py --file urls.txt --ai openrouter
```

## Requirements

| Tool | Install | Notes |
|------|---------|-------|
| Python 3.10+ | `sudo apt install python3` | |
| ffmpeg | `sudo apt install ffmpeg` | |
| yt-dlp | `pip install yt-dlp` | |
| whisper | `pip install openai-whisper` | Auto-detects GPU/CPU |
| json_repair | `pip install json-repair` | |
| ollama | [ollama.com](https://ollama.com) | For local AI (free) |
| openai | `pip install openai` | For ChatGPT/OpenRouter API |
| kokoro | `pip install kokoro soundfile numpy` | For TTS (optional) |
| opencv-python | `pip install opencv-python` | For face tracking (optional) |

### AI Backend (pick one)

| Backend | Cost | Setup |
|---------|------|-------|
| **ollama** | Free (local) | `ollama pull qwen2.5:7b` |
| **chatgpt** | Paid | `export OPENAI_API_KEY="sk-..."` |
| **openrouter** | Free tier | `export OPENROUTER_API_KEY="sk-or-..."` |

## Usage

### Single video

```bash
python ai_clipper.py "https://www.youtube.com/watch?v=..."
```

### Batch

```bash
python ai_clipper.py --file urls.txt
```

### All Options

```
--ai {ollama,chatgpt,openrouter}   AI backend (default: ollama)
--ollama-model MODEL               Ollama model (default: qwen2.5:7b)
--chatgpt-model MODEL              ChatGPT model (default: gpt-4o-mini)
--openai-key KEY                   OpenAI API key
--openrouter-model MODEL           OpenRouter model (default: google/gemini-2.0-flash-001)
--openrouter-key KEY               OpenRouter API key
--clips N                          Number of clips (default: 5)
--whisper-model MODEL              Whisper model (default: small)
--language LANG                    Whisper language (default: auto)
--tts                              Enable TTS voiceover
--tts-voice VOICE                  Kokoro voice (default: af_heart)
--tts-speed SPEED                  TTS speed (default: 1.0)
--face-track                       Face tracking for smarter cropping
--gpu                              GPU hardware encoding (NVENC/AMF/QSV)
--doodle PNG                       Custom doodle overlay
--out DIR                          Output directory (default: outputs)
--continue-on-error                Skip failed URLs
```

### Examples

```bash
# Basic
python ai_clipper.py "URL"

# Full features
python ai_clipper.py "URL" --tts --face-track --gpu

# Batch with ChatGPT
python ai_clipper.py --file urls.txt --ai chatgpt --tts --continue-on-error

# OpenRouter free
python ai_clipper.py "URL" --ai openrouter --openrouter-model meta-llama/llama-3.1-8b-instruct:free
```

## Pipeline

```
YouTube URL
    │
    ▼
┌─────────────┐
│  yt-dlp     │  Download video + metadata
└──────┬──────┘
       ▼
┌─────────────┐
│  Whisper    │  Transcribe → text + SRT + word timestamps
└──────┬──────┘
       ▼
┌─────────────────────────────────────────┐
│  AI (2-step)                            │
│  Step 1: Find 15-30s viral moments      │
│  Step 2: Generate hook + metadata       │
└──────┬──────────────────────────────────┘
       ▼
┌──────────────────────────────────────────────┐
│  ffmpeg (per clip)                           │
│  1. Vertical crop (1080x1920) + blurred bg   │
│     └─ [Optional] Face tracking crop         │
│  2. Title overlay (top)                      │
│  3. HOOK TEXT (big red bar, first 3s)        │
│  4. Subtitles (bottom, from Whisper SRT)     │
│  5. Original audio                           │
│  6. [Optional] TTS voiceover mixed in        │
│  7. [Optional] GPU encoding                   │
└──────────────────────────────────────────────┘
       │
       ▼
  final.mp4  ← Ready to upload to TikTok
```

## Output

```
outputs/
  batch_YYYYMMDD_HHMMSS/
    job_YYYYMMDD_HHMMSS/
      source.mp4
      audio.mp3
      transcript/
      clips/
        clip_01/
          final.mp4           ← Upload this
          notes.json
          caption.txt
          voiceover_script.txt
```

## Features

| Feature | Description | Flag |
|---------|-------------|------|
| **Hook Text** | Big text at start (stops scroll in 3s) | Auto (from AI) |
| **Subtitles** | Indonesian, burned from Whisper SRT | Auto |
| **TTS Voiceover** | Mix with original audio | `--tts` |
| **Face Tracking** | Crops to speaker's face | `--face-track` |
| **GPU Encoding** | NVENC/AMF/QSV with CPU fallback | `--gpu` |
| **15-30s Clips** | Optimal TikTok duration | Auto |
| **Title Overlay** | Auto-shrinks for long titles | Auto |

## Hook System

Each clip starts with a big hook text that stops scroll in 3 seconds:

- Full-width red bar (0.3s - 3s)
- Big white text (56px)
- Shock value / question / curiosity

**AI generates hooks like:**
- "JANGAN SKIP! Rahasia ini bikin..."
- "GILA! Ternyata dia..."
- "1 hal yang gak diketahui orang"

## Face Tracking

When `--face-track` is enabled:
- OpenCV detects speaker's face
- Crops to face instead of center
- Falls back to center if no face

```bash
pip install opencv-python
python ai_clipper.py "URL" --face-track
```

## GPU Acceleration

When `--gpu` is enabled:
- Auto-detects NVENC, AMF, QSV, VideoToolbox
- Falls back to CPU (libx264) on failure
- Handles CUDA OOM gracefully

## TTS Voiceover

When `--tts` is enabled:
- Generates voiceover from commentary_script
- Mixes: original at 30% + TTS at 100%

```bash
# Google TTS (free, native Indonesian) — default
pip install gTTS
python ai_clipper.py "URL" --tts

# Kokoro (offline, English accent)
pip install kokoro soundfile numpy
python ai_clipper.py "URL" --tts --tts-engine kokoro

# Kokoro with GPU
python ai_clipper.py "URL" --tts --tts-engine kokoro --gpu
```

| Engine | Cost | Indonesian | Offline | GPU |
|--------|------|------------|---------|-----|
| **gtts** (default) | Free | Native ✅ | No | No |
| **kokoro** | Free | Accented | Yes | Yes |

## Environment Variables

```bash
# .env file or export directly
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
```

## Files

| File | Purpose |
|------|---------|
| `ai_clipper.py` | Main script |
| `start.sh` | Quick start script |
| `urls.txt.example` | Example URL list |
| `.env.example` | API key template |
| `.gitignore` | Git ignore rules |
| `README.md` | This file |

## License

MIT
