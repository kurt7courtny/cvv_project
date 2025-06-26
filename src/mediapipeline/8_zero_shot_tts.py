# media_pipeline/8_zero_shot_tts.py
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from langflow.custom import Component
from langflow.io import DataFrameInput, Output
from langflow.schema import Data
from textwrap import wrap  # For splitting text into smaller chunks
import soundfile as sf


MAX_TOKENS = 300  # XTTS model token limit


class ZeroShotTTS(Component):
    display_name = "8️⃣ Zero-Shot TTS"
    description = "Coqui XTTS v2"
    icon = "mic"
    name = "ZeroShotTTS"

    inputs = [DataFrameInput(name="df_in", display_name="翻译后表")]
    outputs = [Output(name="tts_out", display_name="TTS 片段", method="synthesize")]

    def synthesize(self) -> Data:
        from TTS.api import TTS  # Import TTS library

        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=True)

        df: pd.DataFrame = self.df_in

        # Validate input DataFrame
        if df.empty or not all(col in df.columns for col in ["wav", "translated", "start", "end"]):
            raise ValueError("Input DataFrame is empty or missing required columns: 'wav', 'translated', 'start', 'end'")

        df = df.dropna(subset=["wav", "translated", "start", "end"])

        pieces: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            ref = Path(row["wav"])
            if not ref.is_file():
                raise FileNotFoundError(f"Reference audio file not found: {ref}")

            # Validate audio file
            with sf.SoundFile(ref) as f:
                duration = len(f) / f.samplerate
                if duration < 1.0:
                    continue

            out_wav = ref.with_suffix(".tts.wav")

            try:
                # Split text into chunks of 400 tokens or fewer
                translated_text = row["translated"]
                text_chunks = wrap(translated_text, MAX_TOKENS)

                chunk_wavs = []

                for i, chunk in enumerate(text_chunks):
                    chunk_out_wav = ref.with_name(f"{ref.stem}_chunk_{i}.tts.wav")
                    tts.tts_to_file(
                        chunk,
                        file_path=str(chunk_out_wav),
                        speaker_wav=str(ref),
                        language="zh",
                    )
                    chunk_wavs.append(str(chunk_out_wav))

                # Combine all chunk WAVs into a single entry
                pieces.append({
                    "start": row["start"],
                    "end": row["end"],
                    "wav": chunk_wavs,
                })

            except RuntimeError as e:
                print(f"Speaker conditioning failed for {ref}, falling back to default speaker.")
                tts.tts_to_file(
                    row["translated"],
                    file_path=str(out_wav),
                    language="zh",
                )
                pieces.append(dict(start=row["start"], end=row["end"], wav=str(out_wav)))

            except Exception as e:
                raise RuntimeError(f"TTS generation failed for row: {row}") from e

        self.status = f"TTS generated {len(pieces)} segments"
        return Data(data={"tts_segments": pieces})