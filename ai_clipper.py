#!/usr/bin/env python3
"""AI Clipper - YouTube-to-TikTok pipeline.

Modules:
- utils.py: Shared utilities
- downloader.py: Video downloading
- transcriber.py: Whisper transcription
- ai_planner.py: AI clip planning
- video_processor.py: Video processing
- tts_engine.py: TTS generation
- subtitles.py: Subtitle handling
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from utils import ensure_tool, hhmmss_to_seconds
from downloader import download_video, read_title
from transcriber import transcribe
from ai_planner import ask_ollama, ask_chatgpt, ask_openrouter
from video_processor import create_clip_package


def process_one_url(args, url):
    """Process a single YouTube URL end-to-end."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(args.out)
    base.mkdir(exist_ok=True)

    workdir = base / f"job_{timestamp}"
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"\n========================================")
    print(f"Processing URL: {url}")
    print(f"Project folder: {workdir}")
    print(f"========================================")

    source_video, info_json = download_video(url, workdir)
    title = read_title(info_json)
    print(f"\nVideo title: {title}")

    audio_path = workdir / "audio.mp3"
    extract_audio(source_video, audio_path)

    transcript_dir = workdir / "transcript"
    transcript_dir.mkdir(exist_ok=True)

    transcript_txt, transcript_srt = transcribe(audio_path, transcript_dir, args.whisper_model, args.language)
    transcript_text = transcript_txt.read_text(encoding="utf-8", errors="ignore")

    if args.ai == "chatgpt":
        ai_plan = ask_chatgpt(transcript_text, args.chatgpt_model, args.clips, "id", api_key=args.openai_key)
    elif args.ai == "openrouter":
        ai_plan = ask_openrouter(transcript_text, args.openrouter_model, args.clips, "id", api_key=args.openrouter_key)
    else:
        ai_plan = ask_ollama(transcript_text, args.ollama_model, args.clips, "id")

    (workdir / "ai_plan.json").write_text(json.dumps(ai_plan, indent=2, ensure_ascii=False), encoding="utf-8")

    clips = ai_plan.get("clips", [])
    if not clips:
        raise RuntimeError("No clips returned by AI.")

    clips_dir = workdir / "clips"
    clips_dir.mkdir(exist_ok=True)

    created = []
    for i, clip in enumerate(clips, start=1):
        try:
            final_clip = create_clip_package(
                i, clip, source_video, clips_dir,
                srt_path=transcript_srt,
                tts_enabled=args.tts,
                tts_voice=args.tts_voice,
                tts_speed=args.tts_speed,
                tts_engine=args.tts_engine,
                face_track=args.face_track,
                use_gpu=args.gpu,
            )
            created.append(final_clip)
        except Exception as e:
            print(f"Failed creating clip {i}: {e}")

    # Write summary
    summary_lines = [
        "# AI Clipper Result", "",
        f"Source URL: {url}",
        f"Source title: {title}",
        f"Project folder: {workdir}", "",
        "## Upload-ready clips", ""
    ]
    for d in created:
        summary_lines.append(f"- {d}")
    summary_lines.extend(["", "## Clip metadata", "",
        "Each clip folder contains:",
        "- `final.mp4` — Ready to upload to TikTok",
        "- `notes.json` — Hook, title, caption, hashtags",
        "- `caption.txt` — Copy-paste caption with hashtags",
        "- `voiceover_script.txt` — Optional voiceover script", ""])
    (workdir / "README.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\nDONE.")
    print(f"Open this folder: {workdir}")

    return {"url": url, "title": title, "folder": str(workdir), "created_clips": [str(d) for d in created]}


def read_urls_from_file(file_path):
    """Read YouTube URLs from a text file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"URL file not found: {file_path}")
    urls = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AI Clipper - YouTube-to-TikTok pipeline")

    # Input
    parser.add_argument("url", nargs="?", help="Single YouTube URL")
    parser.add_argument("--file", help="Text file with YouTube URLs")
    parser.add_argument("--clips", type=int, default=5, help="Number of clips (default: 5)")
    parser.add_argument("--out", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("--continue-on-error", action="store_true", help="Skip failed URLs")

    # AI backend
    parser.add_argument("--ai", choices=["ollama", "chatgpt", "openrouter"], default="ollama")
    parser.add_argument("--ollama-model", default="qwen2.5:7b")
    parser.add_argument("--chatgpt-model", default="gpt-4o-mini")
    parser.add_argument("--openai-key", default=None)
    parser.add_argument("--openrouter-model", default="google/gemini-2.0-flash-001")
    parser.add_argument("--openrouter-key", default=None)

    # Whisper
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--language", default="auto")

    # TTS
    parser.add_argument("--tts", action="store_true", help="Enable TTS voiceover")
    parser.add_argument("--tts-engine", choices=["gtts", "kokoro"], default="gtts")
    parser.add_argument("--tts-voice", default="af_heart")
    parser.add_argument("--tts-speed", type=float, default=1.0)

    # Video
    parser.add_argument("--face-track", action="store_true", help="Face tracking for cropping")
    parser.add_argument("--gpu", action="store_true", help="GPU hardware encoding")
    parser.add_argument("--doodle", default=None)

    args = parser.parse_args()

    ensure_tool("ffmpeg")
    ensure_tool("yt-dlp")
    ensure_tool("whisper")
    if args.ai == "ollama":
        ensure_tool("ollama")

    if not args.url and not args.file:
        print("Please provide a YouTube URL or --file urls.txt")
        sys.exit(1)

    if args.url and args.file:
        print("Use either a single URL or --file, not both.")
        sys.exit(1)

    urls = read_urls_from_file(args.file) if args.file else [args.url]
    if not urls:
        print("No URLs found.")
        sys.exit(1)

    batch_results = []
    batch_started = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(args.out) / f"batch_{batch_started}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBatch folder: {batch_dir}")
    print(f"Total URLs: {len(urls)}")

    for index, url in enumerate(urls, start=1):
        print(f"\n\n========== Video {index}/{len(urls)} ==========")
        original_out = args.out
        try:
            args.out = str(batch_dir)
            result = process_one_url(args, url)
            batch_results.append({"status": "success", **result})
        except Exception as e:
            print(f"\nFAILED: {url}")
            print(f"Error: {e}")
            batch_results.append({"status": "failed", "url": url, "error": str(e)})
            if not args.continue_on_error:
                break
        finally:
            args.out = original_out

    summary_path = batch_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(batch_results, indent=2, ensure_ascii=False), encoding="utf-8")

    markdown_lines = ["# Batch Summary", "", f"Total URLs: {len(urls)}", "", "## Results", ""]
    for item in batch_results:
        if item["status"] == "success":
            markdown_lines.append(f"- SUCCESS: {item['title']}")
            markdown_lines.append(f"  - URL: {item['url']}")
            markdown_lines.append(f"  - Folder: {item['folder']}")
        else:
            markdown_lines.append(f"- FAILED: {item['url']}")
            markdown_lines.append(f"  - Error: {item['error']}")
    (batch_dir / "batch_summary.md").write_text("\n".join(markdown_lines), encoding="utf-8")

    print("\nBATCH DONE.")
    print(f"Summary: {batch_dir / 'batch_summary.md'}")


if __name__ == "__main__":
    main()
