from pathlib import Path
import subprocess

FFMPEG = "ffmpeg"  # Ensure ffmpeg is available

cache_root_path = "/workspace/.cache/"

def run_ffmpeg(cmd: list[str]) -> None:
    """Blocking call to FFmpeg. Raises an exception on error."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"[FFmpeg] {' '.join(cmd)}\n{result.stderr}")