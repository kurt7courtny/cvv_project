# media_pipeline/5_speaker_attribute.py
import torch
from langflow.custom import Component
from langflow.io import DataInput, Output
from langflow.schema import Data


class SpeakerAttribute(Component):
    display_name = "5️⃣ 属性分析"
    description = "情感 + 性别 + 语速"
    icon = "activity"
    name = "SpeakerAttribute"

    inputs = [DataInput(name="asr_in", display_name="转写结果")]
    outputs = [Output(name="attr_out", display_name="属性结果", method="analyze")]

    def analyze(self) -> Data:
        from transformers import pipeline

        emo_pipe = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-emotion",
            top_k=None,
        )

        items = self.asr_in.data["results"]
        for it in items:
            it["emotion"] = emo_pipe(it["text"])[0][0]["label"]
            dur = it["end"] - it["start"]
            it["speed_wps"] = round(len(it["text"].split()) / dur, 2)

        self.status = "属性分析完成"
        return Data(data={"items": items})