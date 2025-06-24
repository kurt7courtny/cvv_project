# media_pipeline/utils.py
from pathlib import Path
import subprocess

FFMPEG = "ffmpeg"  # 请确保 ffmpeg 可用

cache_root_path = "/workspace/.cache/"

def run_ffmpeg(cmd: list[str]) -> None:
    """阻塞式调用 FFmpeg，报错时抛异常。"""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"[FFmpeg] {' '.join(cmd)}\n{result.stderr}")