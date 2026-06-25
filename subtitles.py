"""Subtitle handling: SRT trimming, ASS generation, and burning."""

import re
from pathlib import Path
from utils import run


def trim_srt(srt_path, start_sec, duration, out_path):
    """Trim an SRT file to a specific time range, outputting relative timestamps."""
    if not srt_path or not Path(srt_path).exists():
        return None

    content = srt_path.read_text(encoding="utf-8", errors="ignore")
    end_sec = start_sec + duration

    blocks = re.split(r'\n\n+', content.strip())
    trimmed_blocks = []
    counter = 1

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        ts_match = re.match(
            r'(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})',
            lines[1]
        )
        if not ts_match:
            continue

        h1, m1, s1, ms1, h2, m2, s2, ms2 = ts_match.groups()
        seg_start = int(h1)*3600 + int(m1)*60 + int(s1) + int(ms1)/1000
        seg_end = int(h2)*3600 + int(m2)*60 + int(s2) + int(ms2)/1000

        if seg_end < start_sec or seg_start > end_sec:
            continue

        seg_start = max(seg_start, start_sec)
        seg_end = min(seg_end, end_sec)

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


def generate_capcut_ass(words, output_path):
    """Generate CapCut-style ASS subtitle with word-by-word yellow highlighting."""
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
    chunk_size = 4
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        if not chunk:
            continue
        chunk_start = chunk[0]["start"]
        chunk_end = chunk[-1]["end"]
        line_parts = []
        for j, w in enumerate(chunk):
            word_text = w["word"].strip()
            if not word_text:
                continue
            if j == len(chunk) // 2:
                line_parts.append(f"{{\\c&H00FFFF&\\b1}}{word_text}{{\\c&HFFFFFF&\\b0}}")
            else:
                line_parts.append(word_text)
        line_text = " ".join(line_parts)
        events.append(
            f"Dialogue: 0,{sec_to_ass(chunk_start)},{sec_to_ass(chunk_end)},Default,,0,0,0,,{line_text}"
        )

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


def build_subtitle_filter(srt_path, start_sec, duration):
    """Build ffmpeg subtitle filter string from SRT path."""
    trimmed_srt = trim_srt(srt_path, start_sec, duration, Path(srt_path).parent / "_tmp_sub.srt")

    if trimmed_srt and trimmed_srt.exists():
        srt_abs = str(trimmed_srt.resolve())
        srt_escaped = srt_abs.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
        sub_filter = (
            f"subtitles='{srt_escaped}':"
            "force_style='FontName=Arial,FontSize=14,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H00000000,BorderStyle=3,Outline=1,Shadow=0,"
            "Alignment=2,MarginV=40'"
        )
        print(f"[SUB] Burning subtitles: {srt_abs}")
        return sub_filter, trimmed_srt
    else:
        print(f"[SUB] No subtitles available")
        return None, None
