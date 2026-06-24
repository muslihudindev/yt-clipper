"""TTS voiceover generation using Google TTS or Kokoro."""

from utils import run

try:
    from kokoro import KPipeline
    import soundfile as sf
    import numpy as np
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


def generate_tts(text, output_path, voice="af_heart", speed=1.0, use_gpu=False, engine="gtts"):
    """Generate TTS audio from text.

    Engines:
    - gtts: Online, native Indonesian, free
    - kokoro: Offline, English voice (accented for Indonesian)
    """
    if engine == "gtts":
        return generate_tts_gtts(text, output_path)
    else:
        return generate_tts_kokoro(text, output_path, voice, speed, use_gpu)


def generate_tts_gtts(text, output_path):
    """Generate TTS using Google TTS (free, native Indonesian)."""
    if not GTTS_AVAILABLE:
        raise RuntimeError("gTTS not installed. Run: pip install gTTS")
    print(f"[TTS] Using Google TTS (Indonesian)")
    tts = gTTS(text=text, lang='id', slow=False)
    tts.save(str(output_path))
    return output_path


def generate_tts_kokoro(text, output_path, voice="af_heart", speed=1.0, use_gpu=False):
    """Generate TTS using Kokoro (offline, English voice)."""
    if not KOKORO_AVAILABLE:
        raise RuntimeError("Kokoro not installed. Run: pip install kokoro soundfile")
    device = "cpu"
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                print(f"[TTS] Using GPU: {torch.cuda.get_device_name(0)}")
        except ImportError:
            pass
    print(f"[TTS] Using Kokoro (offline)")
    pipeline = KPipeline(lang_code='a', device=device)
    samples = []
    for gs, ps, audio in pipeline(text, voice=voice, speed=speed):
        samples.append(audio)
    if not samples:
        raise RuntimeError("Kokoro generated no audio")
    audio_out = np.concatenate(samples)
    sf.write(str(output_path), audio_out, 24000)
    return output_path


def mix_audio(original_audio, tts_audio, output_path, original_volume=0.3, tts_volume=1.0):
    """Mix original audio (lowered) with TTS voiceover."""
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
