"""Audio transcription using Whisper."""

import subprocess
from pathlib import Path
from utils import run


def transcribe(audio_path, outdir, whisper_model, language):
    """Transcribe audio using Whisper with GPU auto-detection."""
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
