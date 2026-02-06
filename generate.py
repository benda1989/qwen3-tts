import gradio as gr
import os

BATCH_SIZE = 5

class Generate:
    def __init__(self, save_dir="saved_models"):
        self.model_name = None
        self.model = None
        self.gen_params = {"temperature": 0.7, "top_k": 20, "top_p": 0.95, "repetition_penalty": 1.02}
        self.save_dir = save_dir
        self.speakers = ["my_voice"]

    def get_models(self):
        if not os.path.exists(self.save_dir):
            return []
        models = []
        for name in os.listdir(self.save_dir):
            model_path = os.path.join(self.save_dir, name)
            if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "model.safetensors")):
                models.append(name)
        return models

    def generate(self, **kwargs):
        kwargs.update(self.gen_params)
        if isinstance(kwargs["text"], list):
            kwargs["max_new_tokens"] = max(self.estimate_tokens(t) for t in kwargs["text"])
        else:
            kwargs["max_new_tokens"] = self.estimate_tokens(kwargs["text"])
        try:
            return self.model.generate_custom_voice(**kwargs)
        except Exception as e:
            print(f"生成失败: {e}")
            return None, None

    def estimate_tokens(self, text):
        chinese = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        tokens_per_char = 12 if chinese > len(text) * 0.3 else 8
        return max(200, min(int(len(text) * tokens_per_char * 1.1), 1024))

    def update_buttons(self, text, model, enabled):
        interactive = text and text.strip() and model
        return (
            gr.update(interactive=interactive),
            gr.update(visible=enabled and not interactive)
        )

    def on_model_change(self, model , manager):
        speaker = gr.update(choices=[], value=None)
        if model:
            if model != self.model_name:
                if self.model:
                    del self.model
                self.model_name = model
                self.model = manager.get_model(os.path.join(self.save_dir, model))
            if self.model:
                try:
                    if hasattr(self.model, 'get_supported_speakers'):
                        speakers = self.model.get_supported_speakers()
                        if speakers:
                            self.speakers = speakers
                        elif hasattr(self.model, 'config'):
                            config = self.model.config
                            if hasattr(config, 'talker_config') and config.talker_config.get("spk_id"):
                                self.speakers= list(config.talker_config["spk_id"].keys())
                except:
                    pass
                if self.speakers:
                    speaker = gr.update(choices=self.speakers, value=self.speakers[0])
        return speaker

    def merge_audio(self, audios):
        for item in audios:
            if item.get("audio"):
                yield item["audio"]

    def regenerate_single(self, status, id, text, inst, speaker=None,language=None):
        item = status[id]
        wavs, sr = self.generate(
            text=text,
            speaker=speaker or item["speaker"],
            language=language or item["language"],
            instruct=inst 
        )
        
        if wavs:
            audio = (sr, wavs[0])
            status[id]["audio"] = audio
            status[id]["instruct"] = inst
            return audio
        return None

    def interface(self, manager):
        pass
    def interface2(self, manager):
        pass