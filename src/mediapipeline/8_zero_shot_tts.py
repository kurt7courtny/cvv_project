# media_pipeline/8_zero_shot_tts.py
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from langflow.custom import Component
from langflow.io import DataFrameInput, Output
from langflow.schema import Data


class ZeroShotTTS(Component):
    display_name = "8️⃣ Zero-Shot TTS"
    description = "Coqui XTTS v2"
    icon = "mic"
    name = "ZeroShotTTS"

    inputs = [DataFrameInput(name="df_in", display_name="翻译后表")]
    outputs = [Output(name="tts_out", display_name="TTS 片段", method="synthesize")]

    def synthesize(self) -> Data:
        from TTS.api import TTS

        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)

        df: pd.DataFrame = self.df_in
        pieces: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            ref = row["wav"]
            out_wav = Path(ref).with_suffix(".tts.wav")
            tts.tts_to_file(row["translated"], file_path=str(out_wav), speaker_wav=ref)
            pieces.append(dict(start=row["start"], end=row["end"], wav=str(out_wav)))

        self.status = f"TTS 生成 {len(pieces)} 段"
        return Data(data={"tts_segments": pieces})