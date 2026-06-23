#!/usr/bin/env python3

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from json_repair import repair_json

# Optional: OpenAI SDK (only needed for --ai chatgpt/openrouter)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Optional: Kokoro TTS (only needed for --tts)
try:
    from kokoro import KPipeline
    import soundfile as sf
    import numpy as np
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

# Optional: OpenCV for face tracking (only needed for --face-track)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

def clean_model_output(text):
    """Clean common terminal/model artifacts before JSON parsing."""
    if not text:
        return ""

    # Remove ANSI / terminal escape sequences such as ESC[5D, ESC[K, colors, etc.
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    text = ansi_escape.sub('', text)

    # Remove other control characters but keep normal whitespace.
    text = ''.join(
        ch for ch in text
        if ch in '\n\r\t' or ord(ch) >= 32
    )

    # Remove common markdown fences.
    text = text.replace("```json", "")
    text = text.replace("```", "")

    return text.strip()


def run(cmd, cwd=None, check=True):
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result


def run_capture(cmd, cwd=None, check=True):
    print(f"\n[RUN] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout.strip()


def safe_name(text):
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text.strip())
    return text[:80] or "youtube-video"


def ensure_tool(name):
    if shutil.which(name) is None:
        print(f"Missing tool: {name}")
        sys.exit(1)


def seconds_to_hhmmss(seconds):
    seconds = int(float(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def hhmmss_to_seconds(value):
    value = value.strip()
    parts = value.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(value)


def download_video(url, workdir):
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

    # If yt-dlp already produced source.mp4, do not use the same file as ffmpeg input/output.
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
    try:
        data = json.loads(info_json.read_text())
        return data.get("title", "youtube-video")
    except Exception:
        return "youtube-video"


def generate_tts(text, output_path, voice="af_heart", speed=1.0, lang_code="a"):
    """Generate TTS audio from text using Kokoro."""
    if not KOKORO_AVAILABLE:
        raise RuntimeError("Kokoro not installed. Run: pip install kokoro soundfile")

    pipeline = KPipeline(lang_code=lang_code)
    samples = []

    for gs, ps, audio in pipeline(text, voice=voice, speed=speed):
        samples.append(audio)

    if not samples:
        raise RuntimeError("Kokoro generated no audio")

    audio_out = np.concatenate(samples)
    sf.write(str(output_path), audio_out, 24000)
    return output_path


def mix_audio(original_audio, tts_audio, output_path, original_volume=0.3, tts_volume=1.0):
    """Mix original audio (lowered) with TTS voiceover using ffmpeg."""
    run([
        "ffmpeg", "-y",
        "-i", str(original_audio),
        "-i", str(tts_audio),
        "-filter_complex",
        (
            f"[0:a]volume={original_volume}[orig];"
            f"[1:a]volume={tts_volume}[tts];"
            "[orig][tts]amix=inputs=2:duration=longest[aout]"
        ),
        "-map", "[aout]",
        "-c:a", "aac", "-b:a", "192k",
        str(output_path)
    ])


def extract_audio(video_path, audio_path):
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


def transcribe(audio_path, outdir, whisper_model, language):
    cmd = [
        "whisper",
        str(audio_path),
        "--model", whisper_model,
        "--output_dir", str(outdir),
        "--output_format", "all",
        "--word_timestamps", "True",
    ]

    if language != "auto":
        cmd.extend(["--language", language])

    # Check if CUDA is actually usable
    cuda_ok = False
    try:
        r = subprocess.run(
            ["python", "-c", "import torch; print(torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=10
        )
        cuda_ok = "True" in r.stdout
    except Exception:
        pass

    if not cuda_ok:
        print("[WHISPER] CUDA not available, using CPU")
        cmd.extend(["--device", "cpu"])
        run(cmd)
    else:
        # Try GPU first, fall back to CPU on OOM
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            stderr_text = (result.stderr or "").lower()
            if "out of memory" in stderr_text:
                print("[WARN] GPU OOM — retrying whisper on CPU (slower)...")
                cpu_cmd = cmd + ["--device", "cpu"]
                run(cpu_cmd)
            else:
                print(result.stderr)
                raise RuntimeError(f"Whisper failed: {' '.join(cmd)}")
        else:
            print(result.stderr)

    txt_files = list(outdir.glob("*.txt"))
    srt_files = list(outdir.glob("*.srt"))

    if not txt_files:
        raise RuntimeError("Whisper transcript TXT not found.")

    return txt_files[0], srt_files[0] if srt_files else None


def extract_json_object(text):
    """Extract the largest JSON object from a model response."""
    cleaned = clean_model_output(text)

    # Direct parse path.
    try:
        return json.loads(cleaned), cleaned
    except json.JSONDecodeError:
        pass

    # Repair full output path.
    try:
        repaired = repair_json(cleaned)
        return json.loads(repaired), repaired
    except Exception:
        pass

    # Extract between first { and last }.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("Ollama did not return a JSON-like object.")

    raw_json = cleaned[start:end + 1]
    Path("ollama_extracted_json.txt").write_text(raw_json, encoding="utf-8")

    # Parse extracted object.
    try:
        return json.loads(raw_json), raw_json
    except json.JSONDecodeError:
        pass

    # Repair extracted object. This handles invalid newlines inside strings.
    repaired = repair_json(raw_json)
    return json.loads(repaired), repaired


def normalize_timestamp(value, default):
    """Normalize timestamp-ish values into HH:MM:SS."""
    if value is None:
        return default

    value = str(value).strip()

    # If model returns seconds as number/string.
    if re.fullmatch(r"\d+(\.\d+)?", value):
        return seconds_to_hhmmss(float(value))

    # Convert MM:SS to HH:MM:SS.
    if re.fullmatch(r"\d{1,2}:\d{2}", value):
        return "00:" + value.zfill(5)

    # Accept HH:MM:SS or HH:MM:SS.xxx.
    match = re.search(r"(\d{1,2}):(\d{2}):(\d{2})(?:\.\d+)?", value)
    if match:
        h, m, s = match.groups()
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

    return default


def normalize_clip(clip, index):
    """Accept imperfect Ollama schemas and convert them into the script schema."""
    if not isinstance(clip, dict):
        return None

    fallback_text = (
        clip.get("text")
        or clip.get("description")
        or clip.get("details")
        or clip.get("reason")
        or clip.get("point")
        or ""
    )

    start = normalize_timestamp(
        clip.get("start") or clip.get("start_time") or clip.get("timestamp_start"),
        "00:00:00"
    )
    end = normalize_timestamp(
        clip.get("end") or clip.get("end_time") or clip.get("timestamp_end"),
        "00:00:08"
    )

    # Clamp duration between 5 and 60 seconds.
    try:
        start_sec = hhmmss_to_seconds(start)
        end_sec = hhmmss_to_seconds(end)
        if end_sec <= start_sec:
            end = seconds_to_hhmmss(start_sec + 30)
        elif end_sec - start_sec > 60:
            end = seconds_to_hhmmss(start_sec + 60)
        elif end_sec - start_sec < 5:
            end = seconds_to_hhmmss(start_sec + 30)
    except Exception:
        end = "00:00:30"

    hook = clip.get("hook") or clip.get("tiktok_hook") or "Video ini rame, tapi konteksnya banyak yang miss"
    title = clip.get("title") or clip.get("on_screen_title") or "Konteks Video Viral"

    commentary_script = clip.get("commentary_script") or clip.get("script") or ""
    if not commentary_script:
        if fallback_text:
            commentary_script = (
                "Video ini lagi rame, tapi jangan cuma lihat potongannya saja. "
                f"Konteks singkatnya, {fallback_text} "
                "Menurut gue, video viral seperti ini perlu dicek dari sumber dan versi lengkapnya dulu. "
                "Kalau menurut kamu, ini wajar atau berlebihan?"
            )
        else:
            commentary_script = (
                "Video ini lagi rame, tapi konteksnya banyak yang belum tahu. "
                "Jadi sebelum ikut komentar, lebih aman lihat sumber dan versi lengkapnya dulu. "
                "Menurut gue, potongan video seperti ini bisa gampang bikin salah paham. "
                "Kalau menurut kamu gimana?"
            )

    caption = clip.get("caption") or "Video ini rame, tapi konteksnya perlu dilihat lengkap."
    hashtags = clip.get("hashtags") or ["#fyp", "#viralindonesia", "#trendingindonesia"]
    doodle_ideas = clip.get("doodle_ideas") or ["cute shocked cat", "arrow pointing to key moment", "wait text bubble"]
    risk_note = clip.get("risk_note") or "Pastikan ditambah voiceover/commentary agar tidak terlihat seperti repost mentah."
    clickbait_top = clip.get("clickbait_top") or "PARAH BANGET"
    clickbait_bottom = clip.get("clickbait_bottom") or "LIHAT SAMPE HABIS"

    # Explanation text: array of short lines for overlay
    explanation_text = clip.get("explanation_text") or [
        "Yang bikin rame:",
        "Konteks yang orang miss",
    ]
    if isinstance(explanation_text, str):
        explanation_text = [explanation_text]
    # Ensure max 4 lines, each max 12 words
    explanation_text = [str(line).strip()[:60] for line in explanation_text[:4]]

    # Ending CTA question
    ending_cta = clip.get("ending_cta") or "Menurut kamu ini wajar atau berlebihan?"
    ending_cta = str(ending_cta).strip()[:50]

    if isinstance(hashtags, str):
        hashtags = [tag for tag in hashtags.split() if tag.startswith("#")] or ["#fyp", "#viralindonesia", "#trendingindonesia"]

    if isinstance(doodle_ideas, str):
        doodle_ideas = [doodle_ideas]

    return {
        "start": start,
        "end": end,
        "hook": str(hook).replace("\n", " ").strip(),
        "title": str(title).replace("\n", " ").strip(),
        "explanation_text": explanation_text,
        "ending_cta": ending_cta,
        "commentary_script": str(commentary_script).replace("\n", " ").strip(),
        "caption": str(caption).replace("\n", " ").strip(),
        "hashtags": hashtags,
        "doodle_ideas": doodle_ideas,
        "risk_note": str(risk_note).replace("\n", " ").strip(),
        "clickbait_top": str(clickbait_top).replace("\n", " ").strip()[:20],
        "clickbait_bottom": str(clickbait_bottom).replace("\n", " ").strip()[:20],
    }


def normalize_ai_plan(data, number_of_clips):
    """Normalize correct, wrong, or partial LLM JSON into {'clips': [...]}."""
    if not isinstance(data, dict):
        raise RuntimeError("Ollama JSON root is not an object.")

    # Correct schema.
    if isinstance(data.get("clips"), list):
        raw_clips = data["clips"]

    # Common wrong schemas the model sometimes returns.
    else:
        raw_clips = None
        for key in ["main_points", "moments", "segments", "items", "ideas", "results"]:
            if isinstance(data.get(key), list):
                raw_clips = data[key]
                break

        if raw_clips is None:
            Path("ollama_wrong_schema.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            # Return empty clips instead of crashing — video will be skipped
            print("[WARN] AI returned wrong schema, skipping video. Check ollama_wrong_schema.json")
            return {"clips": []}

    valid_clips = []
    for i, clip in enumerate(raw_clips, start=1):
        normalized = normalize_clip(clip, i)
        if normalized:
            valid_clips.append(normalized)

    if not valid_clips:
        raise RuntimeError("Ollama returned a list, but no valid clip objects.")

    return {"clips": valid_clips[:number_of_clips]}


def build_find_moments_prompt(transcript_text, number_of_clips):
    """Step 1: Ask model to find viral moment timestamps."""
    return f"""Kamu adalah EDITOR SHORT-FORM TIER A untuk konten PODCAST viral (TikTok / Reels / Shorts).

TUGAS: Dari transcript di bawah, HASILKAN TEPAT {number_of_clips} segment viral.

KRITERIA CLIP (WAJIB):
1. DURASI: 15-30 detik SAJA. Tidak boleh lebih.
2. HOOK: Harus ada kalimat yang bikin orang STOP scroll dalam 3 detik pertama.
3. EMOSI: Konflik, kontroversi, humor, pengakuan, statement tajam.
4. PAYOFF: Harus ada ending yang memuaskan (punchline/kesimpulan).

STRONG HOOK RULES:
- TeXt harus BESAR di layar, pertanyaan, shock value
- Contoh hook: "Jangan diskip! Rahasia ini bikin...", "GILA! Ternyata...", "DIAM-diam dia..."
- Hook harus muncul di DETIK PERTAMA video
- Tanpa hook kuat = TikTok algorithm turunkan reach

HINDARI:
- Obrolan filler, basa-basi
- Setup terlalu panjang (>5 detik sebelum payung)
- Clip tanpa payoff/jelas endingnya

ATURAN DURASI (KRITIS):
- Setiap clip 15-30 detik, TIDAK BOLEH lebih
- Hitung durasi dari timestamp transcript
- Jika durasi > 30 detik: PANGKAS bagian tidak relevan

Return ONLY this JSON:
{{"moments": [{{"start":"00:00:00","end":"00:00:20","description":"apa yang terjadi","hook_text":"teks hook yang bikin stop scroll","virality_score":8}}]}}

virality_score: 8-10 (kontroversial/emosional kuat), 5-7 (engaging), 1-4 (biasa)

Transcript:
{transcript_text[:8000]}
"""


def build_clip_metadata_prompt(moment, transcript_text):
    """Step 2: Generate full clip metadata for one moment."""
    return f"""Kamu adalah TikTok clip planner. Hasilkan metadata untuk clip ini.

Momen: {moment.get('start', '00:00:00')} - {moment.get('end', '00:00:20')}
Deskripsi: {moment.get('description', '')}
Virality Score: {moment.get('virality_score', 5)}

Return ONLY this JSON:
{{"hook":"Teks BESAR yang muncul di detik pertama, bikin STOP scroll, maks 10 kata","title":"Apa yang terjadi max 6 kata","clickbait_top":"2-3 kata SHOCK VALUE","clickbait_bottom":"2-3 kata CTA","commentary_script":"Narasi voiceover 2-3 kalimat pendek","caption":"Caption detail 2-3 kalimat,akhiri dengan pertanyaan","hashtags":["#fyp","#viralindonesia"]}}

HOOK RULES (PALING PENTING):
- Hook = teks besar yang muncul di DETIK PERTAMA
- Harus bikin orang STOP scroll dalam 3 detik
- Gunakan: pertanyaan, shock value, angka, kontroversi
- Contoh: "JANGAN SKIP! Ini rahasia...", "GILA! Ternyata...", "DIAM-diam dia...", "1 hal yang gak diketahui orang"
- Maks 10 kata, Bahasa Indonesia casual

CLICKBAIT RULES:
- clickbait_top: SHOCK VALUE, bikin penasaran (contoh: "PARAH INI", "GAK NYANGKA")
- clickbait_bottom: CTA singkat (contoh: "LIAT SAMPE AKHIR", "JANGAN SKIP")

CAPTION RULES:
- Caption = cerita lengkap 2-3 kalimat
- Akhiri dengan pertanyaan untuk engagement
- Contoh: "Guru ini marah besar di kelas. Tapi ternyata alasan dibaliknya bikin netizen debat. Lo setuju?"

Transcript context:
{transcript_text[:4000]}
"""


def ask_ollama(transcript_text, model, number_of_clips, target_language):
    # Step 1: Find moments
    prompt1 = build_find_moments_prompt(transcript_text, number_of_clips)
    print(f"[AI] Step 1: Finding {number_of_clips} viral moments...")

    output1 = run_capture(["ollama", "run", model, prompt1])
    output1 = clean_model_output(output1)
    Path("ollama_step1_output.txt").write_text(output1, encoding="utf-8")

    try:
        moments_data, _ = extract_json_object(output1)
    except Exception as e:
        raise RuntimeError(f"Failed to parse moments: {e}. Check ollama_step1_output.txt")

    moments = moments_data.get("moments", [])
    if not moments:
        print("[WARN] No moments found, saving raw output to ollama_step1_output.txt")
        return {"clips": []}

    print(f"[AI] Found {len(moments)} moments. Step 2: Generating clip metadata...")

    # Step 2: Generate metadata for each moment
    clips = []
    for i, moment in enumerate(moments[:number_of_clips], start=1):
        print(f"[AI] Clip {i}/{min(len(moments), number_of_clips)}: {moment.get('start')} - {moment.get('end')}")
        prompt2 = build_clip_metadata_prompt(moment, transcript_text)

        output2 = run_capture(["ollama", "run", model, prompt2])
        output2 = clean_model_output(output2)

        try:
            clip_data, _ = extract_json_object(output2)
            clip_data["start"] = moment.get("start", "00:00:00")
            clip_data["end"] = moment.get("end", "00:00:30")
            normalized = normalize_clip(clip_data, i)
            if normalized:
                clips.append(normalized)
        except Exception as e:
            print(f"[WARN] Failed to parse clip {i}: {e}")
            continue

    return {"clips": clips}


def ask_chatgpt(transcript_text, model, number_of_clips, target_language, api_key=None):
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai")

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No OpenAI API key. Set OPENAI_API_KEY env var or use --openai-key")

    client = OpenAI(api_key=api_key)
    return _ask_api_two_step(client, model, transcript_text, number_of_clips, "ChatGPT")


def ask_openrouter(transcript_text, model, number_of_clips, target_language, api_key=None):
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai")

    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("No OpenRouter API key. Set OPENROUTER_API_KEY env var or use --openrouter-key")

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    return _ask_api_two_step(client, model, transcript_text, number_of_clips, "OpenRouter")


def _ask_api_two_step(client, model, transcript_text, number_of_clips, label):
    """Shared two-step logic for ChatGPT/OpenRouter."""
    # Step 1: Find moments
    prompt1 = build_find_moments_prompt(transcript_text, number_of_clips)
    print(f"[AI] Step 1 ({label}): Finding {number_of_clips} viral moments...")

    response1 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
            {"role": "user", "content": prompt1}
        ],
        temperature=0.7,
        max_tokens=2000,
    )
    output1 = response1.choices[0].message.content.strip()

    try:
        moments_data, _ = extract_json_object(output1)
    except Exception as e:
        raise RuntimeError(f"Failed to parse moments from {label}: {e}")

    moments = moments_data.get("moments", [])
    if not moments:
        print(f"[WARN] No moments found from {label}")
        return {"clips": []}

    print(f"[AI] Found {len(moments)} moments. Step 2: Generating clip metadata...")

    # Step 2: Generate metadata for each moment
    clips = []
    for i, moment in enumerate(moments[:number_of_clips], start=1):
        print(f"[AI] Clip {i}/{min(len(moments), number_of_clips)}: {moment.get('start')} - {moment.get('end')}")
        prompt2 = build_clip_metadata_prompt(moment, transcript_text)

        response2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                {"role": "user", "content": prompt2}
            ],
            temperature=0.7,
            max_tokens=1500,
        )
        output2 = response2.choices[0].message.content.strip()

        try:
            clip_data, _ = extract_json_object(output2)
            clip_data["start"] = moment.get("start", "00:00:00")
            clip_data["end"] = moment.get("end", "00:00:30")
            normalized = normalize_clip(clip_data, i)
            if normalized:
                clips.append(normalized)
        except Exception as e:
            print(f"[WARN] Failed to parse clip {i}: {e}")
            continue

    return {"clips": clips}


def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 0


def detect_gpu_encoder():
    """Detect available GPU hardware encoder for ffmpeg."""
    encoders = [
        ("h264_nvenc", "NVIDIA NVENC"),
        ("h264_qsv", "Intel Quick Sync"),
        ("h264_amf", "AMD AMF"),
        ("h264_videotoolbox", "Apple VideoToolbox"),
    ]

    for encoder, name in encoders:
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=5
            )
            if encoder in result.stdout:
                print(f"[GPU] Detected: {name} ({encoder})")
                return encoder
        except Exception:
            continue

    print("[GPU] No hardware encoder found, using CPU (libx264)")
    return None


def get_gpu_encoder_args(encoder):
    """Get ffmpeg encoder arguments for the detected GPU."""
    if not encoder:
        return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]

    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "fast", "-cq", "23"]
    elif encoder == "h264_qsv":
        return ["-c:v", "h264_qsv", "-preset", "fast", "-global_quality", "23"]
    elif encoder == "h264_amf":
        return ["-c:v", "h264_amf", "-quality", "speed", "-qp", "23"]
    elif encoder == "h264_videotoolbox":
        return ["-c:v", "h264_videotoolbox", "-q:v", "23"]

    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]


def detect_face_center(video_path, start_sec, duration):
    """Detect the largest face in the video clip and return its center x position (0-1 normalized)."""
    if not CV2_AVAILABLE:
        return None

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(duration * fps)
        start_frame = int(start_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        face_centers = []
        sample_interval = max(1, total_frames // 10)  # Sample 10 frames

        for frame_idx in range(0, total_frames, sample_interval):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))

            if len(faces) > 0:
                # Get largest face
                largest = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest
                center_x = (x + w / 2) / frame.shape[1]  # Normalize to 0-1
                face_centers.append(center_x)

        cap.release()

        if face_centers:
            # Return average face center position
            avg_x = sum(face_centers) / len(face_centers)
            print(f"[FACE] Detected face at x={avg_x:.2f}")
            return avg_x

        return None

    except Exception as e:
        print(f"[FACE] Face detection failed: {e}")
        return None


def make_vertical_clip(source_video, out_path, start, duration, face_track=False, gpu_encoder=None):
    # Validate and clamp timestamps
    video_duration = get_video_duration(source_video)
    start_sec = hhmmss_to_seconds(start)

    if video_duration > 0:
        if start_sec >= video_duration:
            start_sec = max(0, video_duration - duration)
        if start_sec + duration > video_duration:
            duration = video_duration - start_sec
        start = seconds_to_hhmmss(start_sec)

    if duration < 1:
        raise RuntimeError(f"Clip too short or video too short ({video_duration}s)")

    # Face tracking: detect face position for smarter crop
    use_center_crop = True  # Default: center crop
    face_crop_x = None

    if face_track:
        face_x = detect_face_center(source_video, start_sec, duration)
        if face_x is not None:
            # face_x is 0-1 normalized, convert to pixel offset for 1080px width
            face_crop_x = int(face_x * 1080 - 540)  # Offset from center
            face_crop_x = max(-540, min(540, face_crop_x))  # Clamp to valid range
            use_center_crop = False
            print(f"[FACE] Using face-tracking crop offset: {face_crop_x}px")

    # Get encoder args
    encoder_args = get_gpu_encoder_args(gpu_encoder)

    # Build crop filter
    if use_center_crop:
        crop_filter = "crop=1080:1920:(iw-1080)/2:0"
    else:
        # Calculate crop x position centered on face
        crop_filter = f"crop=1080:1920:(iw-1080)/2+{face_crop_x}:0"

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", start,
        "-i", str(source_video),
        "-t", str(duration),
        "-filter_complex",
        (
            f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"{crop_filter},boxblur=18:1[bg];"
            "[0:v]scale=1080:-2[fg];"
            "[bg][fg]overlay=(W-w)/2:(H-h)/2,"
            "fps=30,format=yuv420p[v]"
        ),
        "-map", "[v]",
        "-map", "0:a?",
        *encoder_args,
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-shortest",
        str(out_path)
    ]

    # Try GPU first, fall back to CPU on failure
    result = run(cmd, check=False)
    if result.returncode != 0 and gpu_encoder:
        print(f"[GPU] {gpu_encoder} failed, falling back to CPU (libx264)...")
        cpu_args = get_gpu_encoder_args(None)
        cmdFallback = [
            "ffmpeg",
            "-y",
            "-ss", start,
            "-i", str(source_video),
            "-t", str(duration),
            "-filter_complex",
            (
                f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
                f"{crop_filter},boxblur=18:1[bg];"
                "[0:v]scale=1080:-2[fg];"
                "[bg][fg]overlay=(W-w)/2:(H-h)/2,"
                "fps=30,format=yuv420p[v]"
            ),
            "-map", "[v]",
            "-map", "0:a?",
            *cpu_args,
            "-c:a", "copy",
            "-movflags", "+faststart",
            "-shortest",
            str(out_path)
        ]
        run(cmdFallback)


def generate_capcut_ass(words, output_path):
    """Generate CapCut-style ASS subtitle with word-by-word yellow highlighting.

    words: list of dicts with keys: word, start, end (in seconds)
    """
    # ASS header with CapCut-style formatting
    header = """[Script Info]
Title: CapCut Style Subtitles
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,60,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,50,50,400,1
Style: Highlight,Arial Black,60,&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,50,50,400,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def sec_to_ass(s):
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = int(s % 60)
        cs = int((s % 1) * 100)
        return f"{h}:{m:02d}:{sec:02d}.{cs:02d}"

    events = []
    # Group words into chunks of 4 for display
    chunk_size = 4
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        if not chunk:
            continue

        chunk_start = chunk[0]["start"]
        chunk_end = chunk[-1]["end"]

        # Build the line with highlighted current word
        line_parts = []
        for j, w in enumerate(chunk):
            word_text = w["word"].strip()
            if not word_text:
                continue
            if j == len(chunk) // 2:
                # Highlight the middle word (current word)
                line_parts.append(f"{{\\c&H00FFFF&\\b1}}{word_text}{{\\c&HFFFFFF&\\b0}}")
            else:
                line_parts.append(word_text)

        line_text = " ".join(line_parts)
        events.append(
            f"Dialogue: 0,{sec_to_ass(chunk_start)},{sec_to_ass(chunk_end)},Default,,0,0,0,,{line_text}"
        )

    # Write ASS file
    output_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return output_path


def burn_ass_subtitles(video_path, ass_path, output_path):
    """Burn ASS subtitles into video using ffmpeg."""
    ass_escaped = str(ass_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"ass='{ass_escaped}'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "copy",
        str(output_path)
    ])


def add_hook_overlay(input_video, output_video, hook_text):
    """Add big hook text overlay at the start of the clip (first 3 seconds).

    Hook text appears with:
    - Big white text on red background
    - Centered on screen
    - Animated: appears at 0.5s, stays until 3s
    """
    hook = escape_drawtext(hook_text[:30])

    vf = (
        # Red background box (full width, centered)
        "drawbox=x=0:y=800:w=iw:h=200:color=red@0.9:t=fill:"
        "enable='between(t,0.3,3)',"
        # Big white text
        f"drawtext=text='{hook}':"
        "fontcolor=white:fontsize=56:"
        "x=(w-text_w)/2:y=860:"
        "enable='between(t,0.5,3)'"
    )

    run([
        "ffmpeg",
        "-y",
        "-i", str(input_video),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "copy",
        str(output_video)
    ])


def escape_drawtext(text):
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    text = text.replace("%", "\\%")
    return text


def add_title_overlay(input_video, output_video, title):
    title = escape_drawtext(title[:80])

    # Auto-shrink font: smaller font for longer titles
    title_len = len(title)
    if title_len > 40:
        fontsize = 36
    elif title_len > 25:
        fontsize = 42
    else:
        fontsize = 48

    vf = (
        # Full-width black bar, upper area
        "drawbox=x=0:y=400:w=iw:h=140:color=black@0.7:t=fill,"
        f"drawtext=text='{title}':"
        f"fontcolor=white:fontsize={fontsize}:"
        "x=(w-text_w)/2:y=440:"
        "box=0"
    )

    run([
        "ffmpeg",
        "-y",
        "-i", str(input_video),
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "copy",
        str(output_video)
    ])


def trim_srt(srt_path, start_sec, duration, out_path):
    """Trim an SRT file to a specific time range, outputting relative timestamps."""
    if not srt_path or not Path(srt_path).exists():
        return None

    content = srt_path.read_text(encoding="utf-8", errors="ignore")
    end_sec = start_sec + duration

    # Parse SRT blocks
    blocks = re.split(r'\n\n+', content.strip())
    trimmed_blocks = []
    counter = 1

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        # Parse timestamp line: 00:01:23,456 --> 00:01:25,789
        ts_match = re.match(
            r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})',
            lines[1]
        )
        if not ts_match:
            continue

        h1, m1, s1, ms1, h2, m2, s2, ms2 = ts_match.groups()
        seg_start = int(h1)*3600 + int(m1)*60 + int(s1) + int(ms1)/1000
        seg_end = int(h2)*3600 + int(m2)*60 + int(s2) + int(ms2)/1000

        # Skip if segment is completely outside the clip range
        if seg_end < start_sec or seg_start > end_sec:
            continue

        # Clip to range
        seg_start = max(seg_start, start_sec)
        seg_end = min(seg_end, end_sec)

        # Convert to relative timestamps
        rel_start = seg_start - start_sec
        rel_end = seg_end - start_sec

        def sec_to_srt(s):
            h = int(s // 3600)
            m = int((s % 3600) // 60)
            sec = int(s % 60)
            ms = int((s % 1) * 1000)
            return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

        text = '\n'.join(lines[2:])
        trimmed_blocks.append(f"{counter}\n{sec_to_srt(rel_start)} --> {sec_to_srt(rel_end)}\n{text}")
        counter += 1

    if not trimmed_blocks:
        return None

    out_path.write_text('\n\n'.join(trimmed_blocks) + '\n', encoding="utf-8")
    return out_path


def add_doodle_overlay(input_video, output_video, doodle_path,
                       clickbait_top="PARAH BANGET", clickbait_bottom="LIHAT SAMPE HABIS",
                       explanation_text=None, ending_cta="Menurut kamu ini wajar atau berlebihan?",
                       srt_path=None, clip_start="00:00:00", duration=30):
    # Escape text for ffmpeg drawtext
    top_text = escape_drawtext(clickbait_top.upper()[:20])
    bottom_text = escape_drawtext(clickbait_bottom.upper()[:20])

    # Default explanation lines
    if explanation_text is None:
        explanation_text = ["Yang bikin rame:", "Konteks yang orang miss"]
    if isinstance(explanation_text, str):
        explanation_text = [explanation_text]
    exp_lines = [escape_drawtext(line[:40]) for line in explanation_text[:4]]

    # Escape ending CTA
    cta_text = escape_drawtext(ending_cta[:45])

    # Trim SRT to clip range
    start_sec = hhmmss_to_seconds(clip_start)
    trimmed_srt = trim_srt(srt_path, start_sec, duration, Path(input_video).parent / "_tmp_sub.srt")

    # Debug: check trimmed SRT
    if trimmed_srt and trimmed_srt.exists():
        srt_content = trimmed_srt.read_text(encoding="utf-8")
        srt_lines = [l for l in srt_content.strip().split('\n') if l.strip()]
        print(f"[SUB] Trimmed SRT: {len(srt_lines)} lines, {trimmed_srt.stat().st_size} bytes")
        if len(srt_lines) < 3:
            print(f"[SUB] SRT content: {srt_content[:200]}")
    else:
        print(f"[SUB] No trimmed SRT produced")

    # Build subtitle filter string
    sub_filter = ""
    if trimmed_srt and trimmed_srt.exists():
        # Use absolute path, escape only what ffmpeg needs
        srt_abs = str(trimmed_srt.resolve())
        # ffmpeg subtitles filter needs : and \ escaped
        srt_escaped = srt_abs.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        sub_filter = (
            f"subtitles='{srt_escaped}':"
            "force_style='FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,BorderStyle=3,Outline=2,Shadow=0,"
            "Alignment=2,MarginV=60'"
        )
        print(f"[SUB] Burning subtitles: {srt_abs}")
    else:
        print(f"[SUB] No subtitles available (srt_path={srt_path}, trimmed={trimmed_srt})")

    # --- Build filter_complex ---
    # Layout: subtitles at y=1600, clickbait above them
    # Subtitles at bottom (y~1600), circle+arrow at y~1300, top text at y~160, explanation at y~650

    # No explanation text overlays — clean video
    exp_overlays = []

    all_overlays = "".join(exp_overlays)

    # Determine if we need subtitle filter or just drawtext overlays
    if sub_filter:
        # Two-pass approach: first burn subs, then add overlays
        tmp_sub = Path(input_video).parent / "_tmp_with_subs.mp4"
        run([
            "ffmpeg", "-y",
            "-i", str(input_video),
            "-vf", sub_filter,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "copy",
            str(tmp_sub)
        ])
        # Now apply overlays on top of subtitled video
        _apply_overlays(tmp_sub, output_video, top_text, bottom_text, all_overlays, doodle_path)
        tmp_sub.unlink(missing_ok=True)
    else:
        _apply_overlays(input_video, output_video, top_text, bottom_text, all_overlays, doodle_path)

    # Cleanup temp SRT
    if trimmed_srt and trimmed_srt.exists():
        trimmed_srt.unlink(missing_ok=True)


def _apply_overlays(input_video, output_video, top_text, bottom_text, all_overlays, doodle_path):
    """Apply clean overlays: explanation text + CTA only (subtitles already burned)."""
    if doodle_path and Path(doodle_path).exists():
        # Custom doodle with zoom animation
        run([
            "ffmpeg", "-y",
            "-i", str(input_video),
            "-i", str(doodle_path),
            "-filter_complex",
            (
                "[1:v]scale=350:-1,"
                "zoompan=z='min(zoom+0.002,1.2)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=350x350:fps=30"
                "[d];"
                "[0:v][d]overlay=60:1200:"
                "enable='between(t,0.3,8)'"
                "[v]"
            ),
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "copy",
            str(output_video)
        ])
    else:
        # No overlays — just copy the file
        if all_overlays:
            run([
                "ffmpeg", "-y",
                "-i", str(input_video),
                "-filter_complex",
                all_overlays,
                "-map", "0:v",
                "-map", "0:a?",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-c:a", "copy",
                str(output_video)
            ])
        else:
            shutil.copy(input_video, output_video)


def create_clip_package(index, clip, source_video, clips_dir, doodle_path=None, srt_path=None,
                       tts_enabled=False, tts_voice="af_heart", tts_speed=1.0, face_track=False, use_gpu=False):
    clip_dir = clips_dir / f"clip_{index:02d}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    start = clip.get("start", "00:00:00")
    end = clip.get("end", "00:00:08")

    try:
        duration = max(15, min(30, hhmmss_to_seconds(end) - hhmmss_to_seconds(start)))
    except Exception:
        duration = 20

    # Detect GPU encoder once per clip
    gpu_encoder = detect_gpu_encoder() if use_gpu else None

    # Intermediate files (cleaned up after pipeline)
    tmp_vertical = clip_dir / "_tmp_vertical.mp4"
    tmp_titled = clip_dir / "_tmp_titled.mp4"
    tmp_hooked = clip_dir / "_tmp_hooked.mp4"

    # Final upload-ready output
    final_clip = clip_dir / "final.mp4"

    make_vertical_clip(source_video, tmp_vertical, start, duration, face_track=face_track, gpu_encoder=gpu_encoder)

    # Check vertical clip was created and has content
    if not tmp_vertical.exists() or tmp_vertical.stat().st_size < 1000:
        raise RuntimeError(f"Vertical clip is empty or missing: {tmp_vertical}")

    add_title_overlay(tmp_vertical, tmp_titled, clip.get("title") or clip.get("hook") or "Viral Explained")

    # Check titled clip was created
    if not tmp_titled.exists() or tmp_titled.stat().st_size < 1000:
        raise RuntimeError(f"Titled clip is empty or missing: {tmp_titled}")

    # Add hook overlay (big text at start)
    hook_text = clip.get("hook", "") or clip.get("clickbait_top", "") or "JANGAN SKIP!"
    add_hook_overlay(tmp_titled, tmp_hooked, hook_text)

    add_doodle_overlay(
        tmp_hooked, final_clip, doodle_path,
        clickbait_top=clip.get("clickbait_top", "PARAH BANGET"),
        clickbait_bottom=clip.get("clickbait_bottom", "LIHAT SAMPE HABIS"),
        explanation_text=clip.get("explanation_text", []),
        ending_cta=clip.get("ending_cta", ""),
        srt_path=srt_path,
        clip_start=start,
        duration=duration,
    )

    # Remove intermediates
    tmp_vertical.unlink(missing_ok=True)
    tmp_titled.unlink(missing_ok=True)
    tmp_hooked.unlink(missing_ok=True)

    # Generate TTS voiceover and mix with original audio
    if tts_enabled:
        commentary = clip.get("commentary_script", "")
        print(f"[TTS] Clip {index}: commentary_script length = {len(commentary)}")
        if commentary:
            print(f"[TTS] Generating voiceover for clip {index}...")
            tts_path = clip_dir / "voiceover.wav"
            mixed_path = clip_dir / "_tmp_mixed.mp4"

            try:
                generate_tts(commentary, tts_path, voice=tts_voice, speed=tts_speed)

                # Extract original audio from final clip
                orig_audio = clip_dir / "_tmp_orig_audio.aac"
                run([
                    "ffmpeg", "-y",
                    "-i", str(final_clip),
                    "-vn", "-c:a", "copy",
                    str(orig_audio)
                ])

                # Mix: original at 30% volume + TTS at full volume
                mix_audio(orig_audio, tts_audio=tts_path, output_path=mixed_path,
                         original_volume=0.3, tts_volume=1.0)

                # Replace audio in final clip (use temp file, can't read/write same file)
                tmp_final = clip_dir / "_tmp_final.mp4"
                run([
                    "ffmpeg", "-y",
                    "-i", str(final_clip),
                    "-i", str(mixed_path),
                    "-c:v", "copy",
                    "-map", "0:v:0",
                    "-map", "1:a:0",
                    "-shortest",
                    str(tmp_final)
                ])

                # Swap temp into place
                tmp_final.replace(final_clip)

                # Cleanup
                orig_audio.unlink(missing_ok=True)
                mixed_path.unlink(missing_ok=True)
                tts_path.unlink(missing_ok=True)
                print(f"[TTS] Voiceover mixed for clip {index}")

            except Exception as e:
                print(f"[WARN] TTS failed for clip {index}: {e}")

    # Write metadata
    notes = {
        "source_start": start,
        "source_end": end,
        "hook": clip.get("hook", ""),
        "title": clip.get("title", ""),
        "explanation_text": clip.get("explanation_text", []),
        "ending_cta": clip.get("ending_cta", ""),
        "commentary_script": clip.get("commentary_script", ""),
        "caption": clip.get("caption", ""),
        "hashtags": clip.get("hashtags", []),
        "doodle_ideas": clip.get("doodle_ideas", []),
        "risk_note": clip.get("risk_note", ""),
        "clickbait_top": clip.get("clickbait_top", ""),
        "clickbait_bottom": clip.get("clickbait_bottom", ""),
        "tts_enabled": tts_enabled,
        "file": str(final_clip),
    }

    (clip_dir / "notes.json").write_text(
        json.dumps(notes, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    (clip_dir / "caption.txt").write_text(
        f"{notes['caption']}\n\n{' '.join(notes['hashtags'])}\n",
        encoding="utf-8"
    )

    (clip_dir / "voiceover_script.txt").write_text(
        notes["commentary_script"],
        encoding="utf-8"
    )

    return final_clip


def process_one_url(args, url):
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

    transcript_txt, transcript_srt = transcribe(
        audio_path,
        transcript_dir,
        args.whisper_model,
        args.language
    )

    transcript_text = transcript_txt.read_text(encoding="utf-8", errors="ignore")

    if args.ai == "chatgpt":
        ai_plan = ask_chatgpt(
            transcript_text,
            args.chatgpt_model,
            args.clips,
            "id",
            api_key=args.openai_key,
        )
    elif args.ai == "openrouter":
        ai_plan = ask_openrouter(
            transcript_text,
            args.openrouter_model,
            args.clips,
            "id",
            api_key=args.openrouter_key,
        )
    else:
        ai_plan = ask_ollama(
            transcript_text,
            args.ollama_model,
            args.clips,
            "id"
        )

    (workdir / "ai_plan.json").write_text(
        json.dumps(ai_plan, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    clips = ai_plan.get("clips", [])
    if not clips:
        raise RuntimeError("No clips returned by AI.")

    clips_dir = workdir / "clips"
    clips_dir.mkdir(exist_ok=True)

    created = []
    for i, clip in enumerate(clips, start=1):
        try:
            final_clip = create_clip_package(
                i,
                clip,
                source_video,
                clips_dir,
                args.doodle,
                srt_path=transcript_srt,
                tts_enabled=args.tts,
                tts_voice=args.tts_voice,
                tts_speed=args.tts_speed,
                face_track=args.face_track,
                use_gpu=args.gpu,
            )
            created.append(final_clip)
        except Exception as e:
            print(f"Failed creating clip {i}: {e}")

    summary_lines = [
        "# AI Clipper Result",
        "",
        f"Source URL: {url}",
        f"Source title: {title}",
        f"Project folder: {workdir}",
        "",
        "## Upload-ready clips",
        ""
    ]

    for d in created:
        summary_lines.append(f"- {d}")

    summary_lines.extend([
        "",
        "## Clip metadata",
        "",
        "Each clip folder contains:",
        "- `final.mp4` — Ready to upload to TikTok (vertical, title overlay, original audio)",
        "- `notes.json` — Hook, title, caption, hashtags, commentary script",
        "- `caption.txt` — Copy-paste caption with hashtags",
        "- `voiceover_script.txt` — Optional voiceover script",
        ""
    ])

    (workdir / "README.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\nDONE.")
    print(f"Open this folder: {workdir}")

    return {
        "url": url,
        "title": title,
        "folder": str(workdir),
        "created_clips": [str(d) for d in created]
    }


def read_urls_from_file(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"URL file not found: {file_path}")

    urls = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        urls.append(line)

    return urls


def main():
    parser = argparse.ArgumentParser(
        description="Free Linux AI YouTube-to-TikTok clipper pipeline"
    )

    parser.add_argument("url", nargs="?", help="Single YouTube URL")
    parser.add_argument("--file", help="Text file containing YouTube URLs, one per line")
    parser.add_argument("--clips", type=int, default=5, help="Number of clip ideas per video")
    parser.add_argument("--ai", choices=["ollama", "chatgpt", "openrouter"], default="ollama", help="AI backend: ollama, chatgpt, or openrouter")
    parser.add_argument("--ollama-model", default="qwen2.5:7b", help="Ollama model (default: qwen2.5:7b)")
    parser.add_argument("--chatgpt-model", default="gpt-4o-mini", help="ChatGPT model (default: gpt-4o-mini)")
    parser.add_argument("--openai-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--openrouter-model", default="google/gemini-2.0-flash-001", help="OpenRouter model (default: google/gemini-2.0-flash-001, free)")
    parser.add_argument("--openrouter-key", default=None, help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--whisper-model", default="small", help="Whisper model: tiny/base/small/medium")
    parser.add_argument("--language", default="auto", help="Whisper language: auto/id/en/etc")
    parser.add_argument("--doodle", default=None, help="Optional transparent PNG doodle overlay")
    parser.add_argument("--tts", action="store_true", help="Generate TTS voiceover using Kokoro (pip install kokoro soundfile)")
    parser.add_argument("--tts-voice", default="af_heart", help="Kokoro voice (default: af_heart). Others: af_bella, am_adam, am_michael")
    parser.add_argument("--tts-speed", type=float, default=1.0, help="TTS speech speed (default: 1.0)")
    parser.add_argument("--face-track", action="store_true", help="Use OpenCV face detection for smarter cropping (pip install opencv-python)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU hardware encoding (NVENC/AMF/QSV) when available")
    parser.add_argument("--out", default="outputs", help="Output directory")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue processing next URL if one fails")

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

    if args.file:
        urls = read_urls_from_file(args.file)
    else:
        urls = [args.url]

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
            # Put each job inside the batch folder
            args.out = str(batch_dir)

            result = process_one_url(args, url)
            batch_results.append({
                "status": "success",
                **result
            })

        except Exception as e:
            print(f"\nFAILED: {url}")
            print(f"Error: {e}")

            batch_results.append({
                "status": "failed",
                "url": url,
                "error": str(e)
            })

            if not args.continue_on_error:
                break

        finally:
            args.out = original_out

    summary_path = batch_dir / "batch_summary.json"
    summary_path.write_text(
        json.dumps(batch_results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    markdown_lines = [
        "# Batch Summary",
        "",
        f"Total URLs: {len(urls)}",
        "",
        "## Results",
        ""
    ]

    for item in batch_results:
        if item["status"] == "success":
            markdown_lines.append(f"- SUCCESS: {item['title']}")
            markdown_lines.append(f"  - URL: {item['url']}")
            markdown_lines.append(f"  - Folder: {item['folder']}")
        else:
            markdown_lines.append(f"- FAILED: {item['url']}")
            markdown_lines.append(f"  - Error: {item['error']}")

    (batch_dir / "batch_summary.md").write_text(
        "\n".join(markdown_lines),
        encoding="utf-8"
    )

    print("\nBATCH DONE.")
    print(f"Summary: {batch_dir / 'batch_summary.md'}")

if __name__ == "__main__":
    main()
