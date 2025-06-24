# media_pipeline/6_dialogue_table.py
import pandas as pd
from langflow.custom import Component
from langflow.io import DataInput, Output
from langflow.schema import DataFrame


class DialogueTable(Component):
    display_name = "6️⃣ 对白表"
    description = "生成结构化 DataFrame"
    icon = "table"
    name = "DialogueTable"

    inputs = [DataInput(name="attr_in", display_name="属性结果")]
    outputs = [Output(name="df_out", display_name="对白表", method="build")]

    def build(self) -> DataFrame:
        df = pd.DataFrame(self.attr_in.data["items"])
        self.status = f"行数: {len(df)}"
        return DataFrame(df)