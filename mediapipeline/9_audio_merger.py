# media_pipeline/9_audio_merger.py
from pathlib import Path

from pydub import AudioSegment
from langflow.custom import Component
from langflow.io import DataInput, Output
from langflow.schema import Data


class AudioMerger(Component):
    display_name = "9️⃣ 音频合并"
    description = "按时间戳叠加音轨"
    icon = "layers"
    name = "AudioMerger"

    inputs = [DataInput(name="tts_in", display_name="TTS 片段")]
    outputs = [Output(name="audio_out", display_name="合并音频", method="merge")]

    def merge(self) -> Data:
        segs = sorted(self.tts_in.data["tts_segments"], key=lambda x: x["start"])
        end_ms = int(max(s["end"] for s in segs) * 1000) + 500
        merged = AudioSegment.silent(duration=end_ms)

        for s in segs:
            part = AudioSegment.from_file(s["wav"])
            merged = merged.overlay(part, position=int(s["start"] * 1000))

        out = Path(segs[0]["wav"]).with_name("merged_tts.wav")
        merged.export(out, format="wav")
        self.status = f"输出 {out.name}"
        return Data(data={"merged_audio": str(out)})