# media_pipeline/7_translator.py
import torch
import pandas as pd
from langflow.custom import Component
from langflow.io import DataFrameInput, DropdownInput, Output
from langflow.schema import DataFrame


class Translator(Component):
    display_name = "7️⃣ 翻译"
    description = "Marian-MT 离线翻译"
    icon = "globe"
    name = "Translator"

    inputs = [
        DataFrameInput(name="df_in", display_name="对白表"),
        DropdownInput(
            name="tgt_lang",
            display_name="目标语言",
            options=["zh", "de", "fr", "es"],
            value="zh",
        ),
    ]
    outputs = [Output(name="df_out", display_name="翻译后表", method="translate")]

    def translate(self) -> DataFrame:
        from transformers import MarianMTModel, MarianTokenizer

        df: pd.DataFrame = self.df_in
        src_texts = df["text"].tolist()
        model_name = f"Helsinki-NLP/opus-mt-en-{self.tgt_lang}"
        tok = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        batch = tok(src_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = model.generate(**batch, max_length=512)
        df["translated"] = tok.batch_decode(out, skip_special_tokens=True)
        self.status = f"翻译完成:{df}"
        return DataFrame(df)