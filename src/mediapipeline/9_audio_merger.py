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
        # Sort segments by start time
        segs = sorted(self.tts_in.data["tts_segments"], key=lambda x: x["start"])

        # Calculate total duration of the output audio
        end_ms = int(max(s["end"] for s in segs) * 1000) + 500
        merged = AudioSegment.silent(duration=end_ms)

        for s in segs:
            # Merge all chunks for the current segment into a single audio
            if isinstance(s["wav"], list):
                # Initialize an empty audio segment
                segment_audio = AudioSegment.silent(duration=0)

                for chunk_path in s["wav"]:
                    chunk_audio = AudioSegment.from_file(chunk_path)
                    segment_audio += chunk_audio  # Concatenate chunks

            else:
                # If `wav` is not a list, assume it's a single file path
                segment_audio = AudioSegment.from_file(s["wav"])

            # Overlay the segment audio onto the merged audio timeline
            merged = merged.overlay(segment_audio, position=int(s["start"] * 1000))

        # Export the final merged audio
        out = Path(segs[0]["wav"][0] if isinstance(segs[0]["wav"], list) else segs[0]["wav"]).with_name("merged_tts.wav")
        merged.export(out, format="wav")

        self.status = f"输出 {out.name}"
        return Data(data={"merged_audio": str(out)})