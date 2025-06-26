# media_pipeline/3_speaker_diarization.py
from pathlib import Path
from typing import List, Dict, Any

from langflow.custom import Component
from langflow.io import DataInput, StrInput, Output
from langflow.schema import Data


class SpeakerDiarization(Component):
    display_name = "3️⃣ 说话人分离"
    description = "HuggingFace pyannote"
    icon = "users"
    name = "SpeakerDiarization"

    inputs = [
        DataInput(name="audio_in", display_name="音频路径"),
        StrInput(
            name="hf_token",
            display_name="HF Token(私有模型用)",
            advanced=True,
            value="",
        ),
    ]
    outputs = [Output(name="segments_out", display_name="说话片段", method="diarize")]

    def diarize(self) -> Data:
        from pyannote.audio import Pipeline

        audio_path = Path(self.audio_in.data["audio_path"])
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.hf_token or None,
        ).to(torch.device("cuda"))
        diar = pipeline(str(audio_path))

        segs: List[Dict[str, Any]] = []
        for turn, _, speaker in diar.itertracks(yield_label=True):
            segs.append(
                dict(
                    speaker=speaker,
                    start=round(turn.start, 3),
                    end=round(turn.end, 3),
                )
            )
        self.status = f"片段数: {len(segs)}"
        return Data(data={"segments": segs, "audio_path": str(audio_path)})