# media_pipeline/10_video_composer.py
from pathlib import Path

from langflow.custom import Component
from langflow.io import DataInput, Output
from langflow.schema import Data

from .utils import run_ffmpeg


class VideoComposer(Component):
    display_name = "🔟 视频合成"
    description = "用新音轨替换原视频"
    icon = "film"
    name = "VideoComposer"

    inputs = [
        DataInput(name="video_in", display_name="视频路径"),
        DataInput(name="audio_in", display_name="合并音频"),
    ]
    outputs = [Output(name="video_out", display_name="配音视频", method="compose")]

    def compose(self) -> Data:
        video = Path(self.video_in.data["video_path"])
        audio = Path(self.audio_in.data["merged_audio"])
        out_video = video.with_name(f"{video.stem}_dubbed.mp4")

        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(video),
                "-i",
                str(audio),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-shortest",
                str(out_video),
            ]
        )
        self.status = f"生成 {out_video.name}"
        return Data(data={"dubbed_video": str(out_video)})