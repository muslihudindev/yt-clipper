"""Shared utilities for AI Clipper."""

import re
import subprocess
import shutil
import sys
from pathlib import Path


def clean_model_output(text):
    """Clean common terminal/model artifacts before JSON parsing."""
    if not text:
        return ""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', text)
    text = ''.join(ch for ch in text if ch in '\n\r\t' or ord(ch) >= 32)
    text = text.replace("```json", "").replace("```", "")
    return text.strip()


def run(cmd, cwd=None, check=True):
    """Run a command and optionally raise on failure."""
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def run_capture(cmd, cwd=None, check=True):
    """Run a command and capture stdout."""
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout.strip()


def safe_name(text):
    """Convert text to a safe filename."""
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text.strip())
    return text[:80] or "youtube-video"


def ensure_tool(name):
    """Check if a tool is available in PATH."""
    if shutil.which(name) is None:
        print(f"Missing tool: {name}")
        sys.exit(1)


def seconds_to_hhmmss(seconds):
    """Convert seconds to HH:MM:SS format."""
    seconds = int(float(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def hhmmss_to_seconds(value):
    """Convert HH:MM:SS or similar format to seconds."""
    value = value.strip()
    parts = value.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(value)


def escape_drawtext(text):
    """Escape text for ffmpeg drawtext filter."""
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    text = text.replace("%", "\\%")
    return text
