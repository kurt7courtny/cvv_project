# media_pipeline/2_audio_extractor.py
from pathlib import Path

from langflow.custom import Component
# 导入 FloatInput 以接收浮点数作为输入
from langflow.io import DataInput, FloatInput, Output
from langflow.schema import Data

from .utils import run_ffmpeg


class AudioExtractor(Component):
    display_name = "2️⃣ 音频提取"
    description = "用 FFmpeg 分离音轨"
    icon = "music"
    name = "AudioExtractor"

    inputs = [
        DataInput(name="video_in", display_name="视频路径"),
        # 新增一个浮点数输入项来控制音频截取时长
        StrInput(
            name="duration",
            display_name="截取时长 (秒)",
            info="设置提取音频的时长，单位为秒。如果设置为0或负数，则提取完整音频。",
            value=10,  # 默认值为0，表示提取完整音频
        ),
    ]
    outputs = [Output(name="audio_out", display_name="音频路径", method="extract")]

    def extract(self) -> Data:
        """
        从视频文件中提取音轨，并根据用户设定的时长进行截取。
        """
        video = Path(self.video_in.data["video_path"])
        audio = video.with_suffix(".wav")

        # 基础 FFmpeg 命令
        cmd = [
            "ffmpeg",
            "-y",          # 无需确认，直接覆盖输出文件
            "-i", str(video), # 输入文件
            "-vn",         # 去除视频流
            "-ac", "1",      # 设置音频通道为单声道
            "-ar", "16000",  # 设置采样率为 16000 Hz
        ]

        cmd.extend(["-t", str(self.duration)])

        # 将输出文件路径添加到命令末尾
        cmd.append(str(audio))

        # 执行 FFmpeg 命令
        run_ffmpeg(cmd)

        self.status = f"音频 → {audio.name}"
        return Data(data={"audio_path": str(audio)})