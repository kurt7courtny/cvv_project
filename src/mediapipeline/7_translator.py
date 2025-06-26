import pandas as pd
from langflow.custom import Component
from langflow.io import (
    DataFrameInput,
    DropdownInput,
    Output,
    StrInput,
    BoolInput,
)
from langflow.schema import DataFrame

class PoeTranslator(Component):
    display_name = "7️⃣ Poe API 翻译与润色"
    description = "使用 Poe API 进行文本翻译和对话润色"
    icon = "globe-2"
    name = "PoeTranslator"

    inputs = [
        DataFrameInput(name="df_in", display_name="对白表", required=True),
        StrInput(
            name="api_key",
            display_name="Poe API Key",
            required=True,
            value = ""
        ),
        DropdownInput(
            name="bot_name",
            display_name="Poe Bot 名称",
            options=["Claude-3.5-Sonnet", "GPT-4o", "Assistant", "Claude-3-Opus"],
            value="Claude-3.5-Sonnet",
        ),
        DropdownInput(
            name="tgt_lang",
            display_name="目标语言",
            options=["zh", "de", "fr", "es", "ja", "ko", "ru"],
            value="zh",
        ),
        BoolInput(
            name="enable_polishing",
            display_name="启用润色",
            value=True
        ),
    ]
    outputs = [Output(name="df_out", display_name="翻译后表", method="translate")]

    def translate(self) -> DataFrame:
        """
        Connects to the Poe API to perform translation and optional dialogue polishing.
        """
        import fastapi_poe as fp
        df: pd.DataFrame = self.df_in.copy()
        src_texts = df["text"].tolist()
        api_key = self.api_key
        bot_name = self.bot_name
        enable_polishing = self.enable_polishing

        if not api_key:
            raise ValueError("Poe API Key is required.")

        # A mapping from language code to full language name for clearer prompts.
        lang_map = {
            "zh": "Chinese (Simplified)",
            "de": "German",
            "fr": "French",
            "es": "Spanish",
            "ja": "Japanese",
            "ko": "Korean",
            "ru": "Russian",
        }
        target_language_name = lang_map.get(self.tgt_lang, self.tgt_lang)

        translated_texts = []
        total_rows = len(src_texts)

        for i, text in enumerate(src_texts):
            self.status = f"正在处理: {i + 1}/{total_rows}"
            try:
                # Construct the prompt based on user's choice.
                if enable_polishing:
                    prompt = (
                        f"You are an expert translator and editor. First, remove the sentences that are nothing to do with the context"
                        f"translate the following English dialogue into {target_language_name}. "
                        f"Then, polish the translated text to make it sound perfectly natural, fluent, and authentic for a native speaker. "
                        f"Provide ONLY the final, polished translation, without any explanations or original text.\n\n"
                        f'Original English Text: "{text}"'
                    )
                else:
                    prompt = (
                        f"Translate the following English text to {target_language_name}. "
                        f"Provide only the translation, with no extra text or explanations.\n\n"
                        f'"{text}"'
                    )

                # Prepare the message for the Poe API
                message = fp.ProtocolMessage(role="user", content=prompt)

                # Call the synchronous API endpoint
                response_text = ""
                for partial_response in fp.get_bot_response_sync(
                    messages=[message], bot_name=bot_name, api_key=api_key
                ):
                    response_text += partial_response.text

                translated_texts.append(response_text)

            except Exception as e:
                # Handle API errors or other exceptions
                error_message = f"在处理第 {i + 1} 行时出错: {e}"
                self.status = error_message
                # Append an error message to the results to maintain row alignment
                translated_texts.append(f"ERROR: {e}")
                # Optionally, you might want to stop the process on the first error
                # raise RuntimeError(error_message) from e

        df["translated"] = translated_texts
        self.status = f"翻译完成! 共处理 {total_rows} 行。\n{df}"
        return DataFrame(df)