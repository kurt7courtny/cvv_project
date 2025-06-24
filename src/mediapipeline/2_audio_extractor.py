# media_pipeline/2_audio_extractor.py
from pathlib import Path

from langflow.custom import Component
from langflow.io import DataInput, Output
from langflow.schema import Data

from .utils import run_ffmpeg


class AudioExtractor(Component):
    display_name = "2️⃣ 音频提取"
    description = "用 FFmpeg 分离音轨"
    icon = "music"
    name = "AudioExtractor"

    inputs = [DataInput(name="video_in", display_name="视频路径")]
    outputs = [Output(name="audio_out", display_name="音频路径", method="extract")]

    def extract(self) -> Data:
        video = Path(self.video_in.data["video_path"])
        audio = video.with_suffix(".wav")
        run_ffmpeg(
            ["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-t", "5",  str(audio)]
        )
        self.status = f"音频 → {audio.name}"
        return Data(data={"audio_path": str(audio)})