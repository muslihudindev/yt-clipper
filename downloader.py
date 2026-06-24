"""Video downloading and audio extraction using yt-dlp."""

import json
import shutil
from pathlib import Path
from utils import run


def download_video(url, workdir):
    """Download video from YouTube using yt-dlp."""
    info_json = workdir / "video.info.json"
    output_template = str(workdir / "source.%(ext)s")

    run([
        "yt-dlp",
        "--write-info-json",
        "--no-playlist",
        "--no-check-certificates",
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "--extractor-args", "youtube:player_client=web",
        "-f", "bv*+ba/b",
        "-o", output_template,
        url
    ])

    info_files = list(workdir.glob("source.info.json"))
    if not info_files:
        possible = list(workdir.glob("*.info.json"))
        if possible:
            shutil.move(possible[0], info_json)
    else:
        shutil.move(info_files[0], info_json)

    video_files = [
        p for p in workdir.iterdir()
        if p.is_file()
        and p.name.startswith("source.")
        and not p.name.endswith(".json")
        and p.suffix.lower() in [".mp4", ".webm", ".mkv", ".mov", ".m4v"]
    ]

    if not video_files:
        raise RuntimeError("Downloaded video file not found.")

    source = video_files[0]
    normalized = workdir / "source.mp4"

    if source.resolve() == normalized.resolve():
        original = workdir / "source_original.mp4"
        source.rename(original)
        source = original

    run([
        "ffmpeg",
        "-y",
        "-i", str(source),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(normalized)
    ])

    return normalized, info_json


def read_title(info_json):
    """Read video title from info.json."""
    try:
        data = json.loads(info_json.read_text())
        return data.get("title", "youtube-video")
    except Exception:
        return "youtube-video"


def extract_audio(video_path, audio_path):
    """Extract audio from video as mono 16kHz MP3."""
    run([
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "mp3",
        str(audio_path)
    ])
