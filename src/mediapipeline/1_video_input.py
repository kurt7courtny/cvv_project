# media_pipeline/1_video_input.py
from pathlib import Path

from langflow.custom import Component
from langflow.io import FileInput, Output
from langflow.schema import Data


class VideoInput(Component):
    display_name = "1️⃣ 视频输入"
    description = "选择本地视频文件"
    icon = "upload"
    name = "VideoInput"

    inputs = [
        FileInput(name="video_path", display_name="视频文件", file_types=["mp4", "mov", "mkv"], required=True),
    ]
    outputs = [
        Output(name="video_out", display_name="视频路径", method="send"),
    ]

    def send(self) -> Data:
        path = Path(self.video_path)
        if not path.exists():
            raise FileNotFoundError(path)
        self.status = f"已加载 {path.name}"
        return Data(data={"video_path": str(path)})