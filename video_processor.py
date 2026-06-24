"""Video processing: vertical crop, face tracking, GPU encoding, overlays."""

import shutil
import subprocess
from pathlib import Path
from utils import run, escape_drawtext, hhmmss_to_seconds, seconds_to_hhmmss

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


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
    """Detect the largest face and return its center x position (0-1 normalized)."""
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
        sample_interval = max(1, total_frames // 10)
        for frame_idx in range(0, total_frames, sample_interval):
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))
            if len(faces) > 0:
                largest = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest
                center_x = (x + w / 2) / frame.shape[1]
                face_centers.append(center_x)
        cap.release()
        if face_centers:
            avg_x = sum(face_centers) / len(face_centers)
            print(f"[FACE] Detected face at x={avg_x:.2f}")
            return avg_x
        return None
    except Exception as e:
        print(f"[FACE] Face detection failed: {e}")
        return None


def make_vertical_clip(source_video, out_path, start, duration, face_track=False, gpu_encoder=None):
    """Create vertical clip from source video."""
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

    # Face tracking
    use_center_crop = True
    face_crop_x = None
    if face_track:
        face_x = detect_face_center(source_video, start_sec, duration)
        if face_x is not None:
            face_crop_x = int(face_x * 1080 - 540)
            face_crop_x = max(-540, min(540, face_crop_x))
            use_center_crop = False
            print(f"[FACE] Using face-tracking crop offset: {face_crop_x}px")

    encoder_args = get_gpu_encoder_args(gpu_encoder)

    if use_center_crop:
        crop_filter = "crop=1080:1920:(iw-1080)/2:0"
    else:
        crop_filter = f"crop=1080:1920:(iw-1080)/2+{face_crop_x}:0"

    cmd = [
        "ffmpeg", "-y",
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
            "ffmpeg", "-y",
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


def add_hook_overlay(input_video, output_video, hook_text):
    """Add big hook text overlay at the start of the clip (first 3 seconds)."""
    hook = escape_drawtext(hook_text[:30])
    vf = (
        "drawbox=x=0:y=800:w=iw:h=200:color=red@0.9:t=fill:"
        "enable='between(t,0.3,3)',"
        f"drawtext=text='{hook}':"
        "fontcolor=white:fontsize=56:"
        "x=(w-text_w)/2:y=860:"
        "enable='between(t,0.5,3)'"
    )
    run([
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "copy",
        str(output_video)
    ])


def add_title_overlay(input_video, output_video, title):
    """Add title text overlay at the top of the video."""
    title = escape_drawtext(title[:80])
    title_len = len(title)
    if title_len > 40:
        fontsize = 36
    elif title_len > 25:
        fontsize = 42
    else:
        fontsize = 48
    vf = (
        "drawbox=x=0:y=400:w=iw:h=140:color=black@0.7:t=fill,"
        f"drawtext=text='{title}':"
        f"fontcolor=white:fontsize={fontsize}:"
        "x=(w-text_w)/2:y=440:"
        "box=0"
    )
    run([
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "copy",
        str(output_video)
    ])


def add_doodle_overlay(input_video, output_video, doodle_path, srt_path=None, clip_start="00:00:00", duration=30):
    """Add subtitles to video."""
    from subtitles import build_subtitle_filter

    start_sec = hhmmss_to_seconds(clip_start)
    sub_filter, trimmed_srt = build_subtitle_filter(srt_path, start_sec, duration)

    if sub_filter:
        tmp_sub = Path(input_video).parent / "_tmp_with_subs.mp4"
        run([
            "ffmpeg", "-y",
            "-i", str(input_video),
            "-vf", sub_filter,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "copy",
            str(tmp_sub)
        ])
        shutil.copy(tmp_sub, output_video)
        tmp_sub.unlink(missing_ok=True)
    else:
        shutil.copy(input_video, output_video)

    if trimmed_srt and trimmed_srt.exists():
        trimmed_srt.unlink(missing_ok=True)


def create_clip_package(index, clip, source_video, clips_dir, srt_path=None,
                       tts_enabled=False, tts_voice="af_heart", tts_speed=1.0, tts_engine="gtts",
                       face_track=False, use_gpu=False):
    """Create a complete clip package with all overlays."""
    from tts_engine import generate_tts, mix_audio

    clip_dir = clips_dir / f"clip_{index:02d}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    start = clip.get("start", "00:00:00")
    end = clip.get("end", "00:00:20")

    try:
        duration = max(15, min(30, hhmmss_to_seconds(end) - hhmmss_to_seconds(start)))
    except Exception:
        duration = 20

    gpu_encoder = detect_gpu_encoder() if use_gpu else None

    # Intermediate files
    tmp_vertical = clip_dir / "_tmp_vertical.mp4"
    tmp_titled = clip_dir / "_tmp_titled.mp4"
    tmp_hooked = clip_dir / "_tmp_hooked.mp4"
    final_clip = clip_dir / "final.mp4"

    # Pipeline
    make_vertical_clip(source_video, tmp_vertical, start, duration, face_track=face_track, gpu_encoder=gpu_encoder)
    if not tmp_vertical.exists() or tmp_vertical.stat().st_size < 1000:
        raise RuntimeError(f"Vertical clip is empty or missing: {tmp_vertical}")

    add_title_overlay(tmp_vertical, tmp_titled, clip.get("title") or "Viral Explained")
    if not tmp_titled.exists() or tmp_titled.stat().st_size < 1000:
        raise RuntimeError(f"Titled clip is empty or missing: {tmp_titled}")

    hook_text = clip.get("hook", "") or clip.get("clickbait_top", "") or "JANGAN SKIP!"
    add_hook_overlay(tmp_titled, tmp_hooked, hook_text)

    add_doodle_overlay(tmp_hooked, final_clip, None, srt_path=srt_path, clip_start=start, duration=duration)

    # Cleanup intermediates
    tmp_vertical.unlink(missing_ok=True)
    tmp_titled.unlink(missing_ok=True)
    tmp_hooked.unlink(missing_ok=True)

    # TTS voiceover
    if tts_enabled:
        commentary = clip.get("commentary_script", "")
        if commentary:
            print(f"[TTS] Generating voiceover for clip {index}...")
            tts_path = clip_dir / "voiceover.wav"
            mixed_path = clip_dir / "_tmp_mixed.mp4"
            try:
                generate_tts(commentary, tts_path, voice=tts_voice, speed=tts_speed, use_gpu=use_gpu, engine=tts_engine)
                orig_audio = clip_dir / "_tmp_orig_audio.aac"
                run([
                    "ffmpeg", "-y",
                    "-i", str(final_clip),
                    "-vn", "-c:a", "copy",
                    str(orig_audio)
                ])
                mix_audio(orig_audio, tts_audio=tts_path, output_path=mixed_path, original_volume=0.3, tts_volume=1.0)
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
                tmp_final.replace(final_clip)
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
        "commentary_script": clip.get("commentary_script", ""),
        "caption": clip.get("caption", ""),
        "hashtags": clip.get("hashtags", []),
        "clickbait_top": clip.get("clickbait_top", ""),
        "clickbait_bottom": clip.get("clickbait_bottom", ""),
        "tts_enabled": tts_enabled,
        "file": str(final_clip),
    }
    (clip_dir / "notes.json").write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding="utf-8")
    (clip_dir / "caption.txt").write_text(f"{notes['caption']}\n\n{' '.join(notes['hashtags'])}\n", encoding="utf-8")
    (clip_dir / "voiceover_script.txt").write_text(notes["commentary_script"], encoding="utf-8")

    return final_clip
