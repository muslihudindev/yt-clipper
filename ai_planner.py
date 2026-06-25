"""AI-powered clip planning using Ollama, ChatGPT, or OpenRouter."""

import json
import os
import re
from pathlib import Path
from json_repair import repair_json
from utils import clean_model_output, run_capture

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def extract_json_object(text):
    """Extract the largest JSON object from a model response."""
    cleaned = clean_model_output(text)
    try:
        return json.loads(cleaned), cleaned
    except json.JSONDecodeError:
        pass
    try:
        repaired = repair_json(cleaned)
        return json.loads(repaired), repaired
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("Ollama did not return a JSON-like object.")
    raw_json = cleaned[start:end + 1]
    Path("ollama_extracted_json.txt").write_text(raw_json, encoding="utf-8")
    try:
        return json.loads(raw_json), raw_json
    except json.JSONDecodeError:
        pass
    repaired = repair_json(raw_json)
    return json.loads(repaired), repaired


def normalize_timestamp(value, default):
    """Normalize timestamp values into HH:MM:SS."""
    if value is None:
        return default
    value = str(value).strip()
    if re.fullmatch(r"\d+(\.\d+)?", value):
        from utils import seconds_to_hhmmss
        return seconds_to_hhmmss(float(value))
    if re.fullmatch(r"\d{1,2}:\d{2}", value):
        return "00:" + value.zfill(5)
    match = re.search(r"(\d{1,2}):(\d{2}):(\d{2})(?:\.\d+)?", value)
    if match:
        h, m, s = match.groups()
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"
    return default


def normalize_clip(clip, index):
    """Normalize imperfect LLM output into the expected clip schema."""
    if not isinstance(clip, dict):
        return None

    fallback_text = (
        clip.get("text") or clip.get("description") or clip.get("details")
        or clip.get("reason") or clip.get("point") or ""
    )

    start = normalize_timestamp(
        clip.get("start") or clip.get("start_time") or clip.get("timestamp_start"),
        "00:00:00"
    )
    end = normalize_timestamp(
        clip.get("end") or clip.get("end_time") or clip.get("timestamp_end"),
        "00:00:20"
    )

    from utils import hhmmss_to_seconds, seconds_to_hhmmss
    try:
        start_sec = hhmmss_to_seconds(start)
        end_sec = hhmmss_to_seconds(end)
        if end_sec <= start_sec:
            end = seconds_to_hhmmss(start_sec + 20)
        elif end_sec - start_sec > 30:
            end = seconds_to_hhmmss(start_sec + 30)
        elif end_sec - start_sec < 15:
            end = seconds_to_hhmmss(start_sec + 20)
    except Exception:
        end = "00:00:20"

    hook = clip.get("hook") or clip.get("tiktok_hook") or "JANGAN SKIP!"
    title = clip.get("title") or clip.get("on_screen_title") or "Konteks Video Viral"

    commentary_script = clip.get("commentary_script") or clip.get("script") or ""
    if not commentary_script:
        if fallback_text:
            commentary_script = f"Video ini lagi rame, {fallback_text}"
        else:
            commentary_script = "Video ini lagi rame, tapi konteksnya banyak yang belum tahu."

    caption = clip.get("caption") or "Video ini rame, tapi konteksnya perlu dilihat lengkap."
    hashtags = clip.get("hashtags") or ["#fyp", "#viralindonesia", "#trendingindonesia"]
    clickbait_top = clip.get("clickbait_top") or "PARAH BANGET"
    clickbait_bottom = clip.get("clickbait_bottom") or "LIHAT SAMPE HABIS"

    if isinstance(hashtags, str):
        hashtags = [tag for tag in hashtags.split() if tag.startswith("#")] or ["#fyp", "#viralindonesia", "#trendingindonesia"]

    return {
        "start": start,
        "end": end,
        "hook": str(hook).replace("\n", " ").strip(),
        "title": str(title).replace("\n", " ").strip(),
        "commentary_script": str(commentary_script).replace("\n", " ").strip(),
        "caption": str(caption).replace("\n", " ").strip(),
        "hashtags": hashtags,
        "clickbait_top": str(clickbait_top).replace("\n", " ").strip()[:20],
        "clickbait_bottom": str(clickbait_bottom).replace("\n", " ").strip()[:20],
    }


def normalize_ai_plan(data, number_of_clips):
    """Normalize LLM output into {'clips': [...]} format."""
    if not isinstance(data, dict):
        raise RuntimeError("Ollama JSON root is not an object.")

    if isinstance(data.get("clips"), list):
        raw_clips = data["clips"]
    else:
        raw_clips = None
        for key in ["main_points", "moments", "segments", "items", "ideas", "results"]:
            if isinstance(data.get(key), list):
                raw_clips = data[key]
                break
        if raw_clips is None:
            Path("ollama_wrong_schema.json").write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print("[WARN] AI returned wrong schema, skipping video.")
            return {"clips": []}

    valid_clips = []
    for i, clip in enumerate(raw_clips, start=1):
        normalized = normalize_clip(clip, i)
        if normalized:
            valid_clips.append(normalized)

    return {"clips": valid_clips[:number_of_clips]}


def build_find_moments_prompt(transcript_text, number_of_clips):
    """Build Step 1 prompt: find viral moment timestamps with scoring."""
    return f"""You are an expert short-form video editor trained on millions of high-performing TikTok, Reels, and Shorts.

TASK: Find {number_of_clips} viral moments from this transcript. Each moment MUST score >70 virality.

VIRAL MOMENT DETECTION:
Analyze for:
1. STRONG HOOK (0-100): Surprising statement, contrarian opinion, secret revealed, mistake exposed, pain point, controversial claim, shocking statistic, curiosity gap
2. EMOTIONAL PEAKS (0-100): Excitement, anger, fear, inspiration, humor, tension, surprise
3. AUDIENCE VALUE (0-100): Actionable advice, frameworks, lessons, personal stories
4. NARRATIVE COMPLETENESS (0-100): Must have context + main point + conclusion
5. RETENTION POTENTIAL (0-100): Watch-through, rewatch, share, comment probability

VIRALITY SCORING FORMULA:
ViralityScore = (0.30 × HookStrength + 0.20 × EmotionalImpact + 0.20 × AudienceValue + 0.15 × RetentionPotential + 0.10 × NarrativeCompleteness)

CLIP SELECTION RULES:
- ONLY select moments with virality_score > 70
- Hook score > 75
- Narrative completeness > 65
- Duration: 15-30 seconds
- Must include: setup → climax → payoff
- NEVER cut mid-sentence

AVOID:
- Rambling sections
- Repeated information
- Long introductions (>5s before payoff)
- Dead air, filler, transitions

Return ONLY this JSON:
{{"moments": [{{"start":"00:00:00","end":"00:00:20","description":"apa yang terjadi","hook_text":"teks hook yang bikin stop scroll","virality_score":85,"hook_score":90,"emotion_score":80,"value_score":75,"reasoning":["reason 1","reason 2"]}}]}}

Transcript:
{transcript_text[:8000]}
"""


def build_clip_metadata_prompt(moment, transcript_text):
    """Build Step 2 prompt: generate specific, content-grounded metadata."""
    return f"""You are an expert short-form video editor. Generate metadata that SPECIFICALLY describes what happens in this clip.

CLIP CONTEXT:
- Start: {moment.get('start', '00:00:00')}
- End: {moment.get('end', '00:00:20')}
- What happens: {moment.get('description', '')}
- Virality Score: {moment.get('virality_score', 75)}

Return ONLY this JSON:
{{"hook":"Opening line from the clip (what was actually said), max 10 words","title":"What SPECIFICALLY happens in this clip, max 6 words","clickbait_top":"2-3 words about the specific moment","clickbait_bottom":"2-3 words CTA","commentary_script":"2-3 sentence voiceover summarizing this specific moment","caption":"What happens in this clip + why it matters, 2-3 sentences","hashtags":["#fyp","#viralindonesia"]}}

TITLE RULES (CRITICAL - MUST BE SPECIFIC):
- Title = what ACTUALLY happens in this specific clip
- NOT generic clickbait. DESCRIBE THE MOMENT.
- BAD: "GILA! Video Viral!!!" (generic)
- GOOD: "Guru Marah Besar di Kelas" (describes what happens)
- GOOD: "Dia Ngaku Salah di Depan Semua" (describes the moment)
- GOOD: "Netizen Debat Keras Soal Ini" (describes the reaction)
- Match the title to the transcript content

HOOK RULES:
- Hook = what was actually said at the start of the clip
- Use the speaker's actual words or a summary
- Must be specific to this moment, not generic

CAPTION RULES:
- Describe what happens in the clip (specific moment)
- Explain why it matters or why it went viral
- End with a question for engagement
- Example: "Guru ini marah besar karena siswa ketiduran. Tapi alasannya bikin netizen debat. Lo setuju?"

TRANSCRIPT EXCERPT (for context):
{transcript_text[:4000]}
"""


def ask_ollama(transcript_text, model, number_of_clips, target_language):
    """Find viral moments using Ollama (2-step process)."""
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
    """Find viral moments using ChatGPT (2-step process)."""
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed. Run: pip install openai")
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No OpenAI API key. Set OPENAI_API_KEY env var or use --openai-key")
    client = OpenAI(api_key=api_key)
    return _ask_api_two_step(client, model, transcript_text, number_of_clips, "ChatGPT")


def ask_openrouter(transcript_text, model, number_of_clips, target_language, api_key=None):
    """Find viral moments using OpenRouter (2-step process)."""
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
        temperature=0.7, max_tokens=2000,
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
            temperature=0.7, max_tokens=1500,
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
