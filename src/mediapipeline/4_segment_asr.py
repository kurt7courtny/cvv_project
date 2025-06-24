# media_pipeline/4_segment_asr.py
from pathlib import Path
from typing import List, Dict, Any

import torch
from langflow.custom import Component
from langflow.io import DataInput, Output
from langflow.schema import Data

from .utils import run_ffmpeg


class SegmentASR(Component):
    display_name = "4️⃣ 语音识别"
    description = "Whisper (HF pipeline)"
    icon = "type"
    name = "SegmentASR"

    inputs = [DataInput(name="seg_in", display_name="说话片段")]
    outputs = [Output(name="asr_out", display_name="转写结果", method="transcribe")]

    def transcribe(self) -> Data:
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            chunk_length_s=30,
            device=device,
        )

        segs: List[Dict[str, Any]] = self.seg_in.data["segments"]
        audio_path = Path(self.seg_in.data["audio_path"])

        results = []
        for seg in segs:
            wav_seg = audio_path.with_name(
                f"{audio_path.stem}_{seg['start']:.2f}_{seg['end']:.2f}.wav"
            )
            run_ffmpeg(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-ss",
                    str(seg["start"]),
                    "-to",
                    str(seg["end"]),
                    str(wav_seg),
                ]
            )
            text = asr(str(wav_seg))["text"].strip()
            results.append({**seg, "text": text, "wav": str(wav_seg)})

        self.status = f"ASR 完成 {len(results)} 段"
        return Data(data={"results": results})