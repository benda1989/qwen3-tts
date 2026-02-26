import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import tempfile
import shutil
import subprocess
import threading
from datetime import datetime

import torch
import torchaudio
import gradio as gr
from qwen_tts import Qwen3TTSModel

SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

def default_device():
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"
def get_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(f"cuda:{i}")
    return devices

class UI:
    LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"]
    DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    def __init__(self):
        self.save_dir = "models"
        self.device = default_device()
        self.asr_model = None
        self.logs = ""
        self.status = "Done"
        self.dtype = self.DTYPES["bfloat16"]

    def set_device(self, device):
        self.device = device
        self.init_asr()

    def set_dtype(self, dtype):
        self.dtype = self.DTYPES[dtype]

    def init_asr(self):
        if self.asr_model is not None:
            del self.asr_model 
        try:
            from funasr import AutoModel
            self.asr_model = AutoModel(
                model="iic/SenseVoiceSmall",
                device=self.device,
                disable_update=True
            )
        except Exception as e:
            print(f"ASR load failed: {e}")
    
    def recognize(self, audio_path):
        if not self.asr_model:
            self.init_asr()
        try:
            result = self.asr_model.generate(input=audio_path, language="auto", use_itn=False )
            return result[0]["text"].split('|>')[-1].strip() if result else ""
        except:
            return ""

    def log(self, message, status="Processing"):
        print(message)
        self.logs += message + "\n"
        self.status = status
    
    def prepare_training_data(self, audio_files, transcripts, ref_audio=None):
        temp_dir = tempfile.mkdtemp(prefix="qt_training_")
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        jsonl_path = os.path.join(temp_dir, "train_raw.jsonl")
        ref_audio_path = None
        def c24(ap,dp):
            waveform, sr = torchaudio.load(ap)
            if sr != 24000:
                waveform = torchaudio.functional.resample(waveform, sr, 24000)
            torchaudio.save(dp, waveform, 24000)
            return dp
        if ref_audio:
            ref_audio_path = c24(ref_audio, os.path.join(data_dir, "ref.wav"))
            self.log(f"Using uploaded reference audio")

        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for i, (audio_file, transcript) in enumerate(zip(audio_files, transcripts)):
                if audio_file is None or not transcript.strip():
                    continue
                audio_filename = f"utt{i:04d}.wav"
                audio_dest = c24(audio_file.name if hasattr(audio_file, 'name') else audio_file, os.path.join(data_dir, audio_filename))
                if ref_audio_path is None:
                    ref_audio_path = audio_dest
                    self.log(f"Using {audio_filename} as reference audio for all samples")

                entry = {
                    "audio": audio_dest,
                    "text": transcript.strip(),
                    "ref_audio": ref_audio_path 
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        self.log(f"Prepared {len(os.listdir(data_dir))} training samples with audio codes")
        return temp_dir
    
    def train(self, audio_files, texts, ref_audio, model_name, speaker, base_model, lr, batch_size, epochs):
        if not audio_files or not texts:
            self.log("❌ Audio files or texts are empty!", "Done")
            return
        self.log("📝 Preparing data with audio codes...")
        training_dir = self.prepare_training_data(audio_files, texts, ref_audio) 
        self.log(f"Cache dir: {training_dir}")
        prepare_cmd = [
            "python", "finetuning/prepare_data.py",
            "--tokenizer_model_path", "Qwen3-TTS-Tokenizer-12Hz",
            "--input_jsonl", os.path.join(training_dir, "train_raw.jsonl"),
            "--output_jsonl", os.path.join(training_dir, "train_with_codes.jsonl")
        ]
        
        env = os.environ.copy()
        project_root = os.path.dirname(os.path.abspath(__file__))
        env['PYTHONPATH'] = project_root + ":" + env.get('PYTHONPATH', '')
        env['CUDA_VISIBLE_DEVICES']=self.device.split(":")[-1]
        if self.device.startswith("cuda:"):
            env['CUDA_VISIBLE_DEVICES'] = self.device.split(":")[-1]
        else:
            print(self.device)
        result = subprocess.run(prepare_cmd, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            self.log(f"❌ Data preparation failed:" )
            self.log(result.stderr, "Done")
            shutil.rmtree(training_dir)
            return 
        self.log("✅ Data preparation completed successfully!" )

        try:
            output_dir = os.path.join(self.save_dir, model_name)
            os.makedirs(output_dir, exist_ok=True)
            train_cmd = [
                "python", "finetuning/sft_12hz.py",
                "--init_model_path", base_model,
                "--output_model_path", output_dir,
                "--train_jsonl", os.path.join(training_dir, "train_with_codes.jsonl"),
                "--batch_size", str(batch_size),
                "--lr", str(lr),
                "--num_epochs", str(epochs),
                "--speaker_name", speaker,
            ]
            self.log("🔥 Starting training process..." )
            process = subprocess.Popen(
                train_cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True, 
                env=env, 
                bufsize=1
            )
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                self.log(line)            
            process.wait()
            
            if process.returncode == 0:
                final_path = os.path.join(output_dir, f"checkpoint-epoch-{epochs-1}")
                if os.path.exists(final_path):
                    self.log("✅ Train successfully! Ready for testing.")
                else:
                    self.log(f"⚠️ Model checkpoint not found at: {final_path} ")
            else:
                self.log("❌ TRAINING FAILED!")
        except Exception as e:
            self.log(f"❌ Training error: {str(e)}" )
        finally:
            if training_dir and os.path.exists(training_dir):
                shutil.rmtree(training_dir)
                self.log("🧹 Cleaned up temporary training files" )
            self.status = "Done"

    def get_information(self):
        return self.status, self.logs
    
    def get_models(self):
        if not os.path.exists(self.save_dir):
            return []
        return [name for name in os.listdir(self.save_dir) if os.path.isdir(os.path.join(self.save_dir, name))]
    
    def get_model(self, model_path):
        try:
            return Qwen3TTSModel.from_pretrained(
                    model_path,
                    device_map=self.device,
                    torch_dtype=self.dtype,
                    dtype=self.dtype,
                    attn_implementation="flash_attention_2" if "cuda" in self.device else "eager"
                )
        except:
            return None
    def test_batch(self, model_name, text, language, speaker, instruct):
        if not model_name:
            return [], "Please select a model"
        model_dir = os.path.join(self.save_dir, model_name)
        if not os.path.exists(model_dir):
            return [], f"Model directory not found: {model_dir}"
        checkpoints = []
        for item in sorted(os.listdir(model_dir)):
            if item.startswith("checkpoint-"):
                checkpoint_path = os.path.join(model_dir, item)
                if os.path.exists(os.path.join(checkpoint_path, "model.safetensors")):
                    checkpoints.append((item, checkpoint_path))
        if not checkpoints:
            return [], "No checkpoints found"
        results = []
        for checkpoint_name, checkpoint_path in checkpoints:
            try:
                model = self.get_model(checkpoint_path)
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language=language,
                    speaker=speaker,
                    instruct=instruct if instruct else None
                )
                results.append((sr, wavs[0], checkpoint_name, checkpoint_path))
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as e:
                continue        
        return results, "Done"
    
    def save_model_checkpoint(self, checkpoint, name):
        """保存模型检查点到固定位置"""
        if not name:
            return "请输入保存名称"
        
        save_path = os.path.join(SAVE_DIR, name)
        if os.path.exists(save_path):
            return f"名称 '{name}' 已存在，请使用其他名称"
        
        try:
            shutil.copytree(checkpoint, save_path)
            return f"✅ 模型已保存为: {name}"
        except Exception as e:
            return f"❌ 保存失败: {str(e)}"


def create_ui(train=False,device=None,dtype=None):
    ui = UI()
    with gr.Blocks(title="Qwen3-TTS Train benda1989") as interface:
        gr.Markdown("# 🎙️ Qwen3-TTS 语音合成与训练")
        if device is None or dtype is None:
            with gr.Row():
                if device is None:
                    device_dropdown = gr.Dropdown(
                        choices=get_devices(),
                        value=default_device(),
                        label="🖥️ 设备",
                        scale=1
                    )
                    device_dropdown.change(lambda d: ui.set_device(d), inputs=[device_dropdown])
                else:
                    gr.Markdown(f"🖥️ **设备**: {device}")
                if dtype is None:
                    dtype_dropdown = gr.Dropdown(
                        choices=list(ui.DTYPES.keys()),
                        value="bfloat16",
                        label="🔢 数据类型",
                        scale=1
                    )
                    dtype_dropdown.change(lambda dt: ui.set_dtype(dt), inputs=[dtype_dropdown])
                else:
                    gr.Markdown(f"🔢 **数据类型**: {dtype}")
        with gr.Tabs():  
            with gr.Tab("Generate"):      
                from generate import Generate
                generate = Generate(SAVE_DIR)
                generate.interface(ui)
            with gr.Tab("Generates"):
                generate.interface2(ui)
            if not train:
                return interface
            with gr.Tab("Train"):

                with gr.Tab("📁 1 Prepare"):
                    with gr.Column() as upload_area:
                        files = gr.File(file_count="multiple", file_types=["audio"], label="Audio Files")
                    data_state = gr.State([])

                    with gr.Row():
                        add_more_btn = gr.UploadButton("Add More", file_count="multiple", file_types=["audio"], visible=False)
                        clear_btn = gr.Button("Clear All", variant="stop", visible=False)

                    @gr.render(inputs=[data_state], triggers=[data_state.change])
                    def render_items(data_list):
                        if not data_list:
                            gr.Markdown("*No files uploaded*")
                            return

                        for idx, item in enumerate(data_list):
                            path = item["path"]
                            name = os.path.basename(path)

                            with gr.Row():
                                audio = gr.Audio(sources='upload', type='filepath', value=path, label=name, interactive=True)
                                text = gr.Textbox(value=item["text"], label="", lines=5)

                                def on_audio_edit(new_path, id=idx):
                                    if not new_path:
                                        del data_list[id]
                                        return data_list, gr.update()
                                    if new_path != data_list[id]["path"]:
                                        data_list[id]["path"] = new_path
                                        new_text = ui.recognize(new_path)
                                        data_list[id]["text"] = new_text
                                        return data_list, new_text
                                    return data_list, gr.update()

                                audio.change(on_audio_edit, inputs=[audio], outputs=[data_state, text])

                                def update_text(new_text, id=idx):
                                    data_list[id]["text"] = new_text

                                text.change(update_text, inputs=[text])

                    def audio_duration(path):
                        try:
                            info = torchaudio.info(path)
                            return info.num_frames / info.sample_rate
                        except Exception as e:
                            try:
                                waveform, sr = torchaudio.load(path)
                                return waveform.shape[1] / sr
                            except Exception as e2:
                                return 0

                    def process_files(uploaded, prev_data=None):
                        if not uploaded:
                            return [], gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
                        prev_texts = {item["path"]: item["text"] for item in prev_data} if prev_data else {}
                        data = []
                        for file in uploaded:
                            path = file.name if hasattr(file, 'name') else file
                            dur = audio_duration(path)
                            if dur < 5:
                                print(f"Skip {path} with duration {dur:.2f} seconds")
                                continue
                            text = prev_texts.get(path, ui.recognize(path))
                            data.append({"path": path, "text": text, "duration": dur})
                        data.sort(key=lambda x: x["duration"])
                        return data, gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)

                    files.change(process_files, [files, data_state], [data_state, add_more_btn, clear_btn, upload_area])

                    def add_more_files(old_data, new_files):
                        existing_paths = {item["path"] for item in old_data} if old_data else set()
                        data = list(old_data) if old_data else []
                        for file in new_files:
                            path = file.name if hasattr(file, 'name') else file
                            if path not in existing_paths:
                                dur = audio_duration(path)
                                if dur < 3:
                                    continue
                                data.append({"path": path, "text": ui.recognize(path), "duration": dur})
                        data.sort(key=lambda x: x["duration"])
                        return data, gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)

                    add_more_btn.upload(
                        add_more_files,
                        [data_state, add_more_btn],
                        [data_state, add_more_btn, clear_btn, upload_area]
                    )

                    clear_btn.click(
                        lambda: ([], gr.update(value=None), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)),
                        outputs=[data_state, files, add_more_btn, clear_btn, upload_area]
                    )

                with gr.Tab("⚙️ 2 Training"):
                    with gr.Row():
                        model_name = gr.Textbox( label="Model Name", value=f"custom_{datetime.now().strftime('%y%m%d_%H%M')}" )
                        speaker_name = gr.Textbox(label="Speaker Name", value="my_voice")

                    with gr.Row():
                        ref_audio = gr.Audio(sources='upload', type='filepath', label="Reference Audio (optional, defaults to first training sample)")
                        base_model = gr.Dropdown( choices=["Qwen3-TTS-12Hz-1.7B-Base", "Qwen3-TTS-12Hz-0.6B-Base"], value="Qwen3-TTS-12Hz-1.7B-Base", label="Base Model")
                    gr.Markdown("⚠️ 30个音频，6-10s，建议超参：1e-5 / 2 / 4，loss低于7效果会变差")
                    with gr.Row():
                        lr = gr.Slider(1e-6, 1e-3, value=1e-5, label="Learning Rate")
                        batch_size = gr.Slider(1, 16, value=2, step=1, label="Batch Size")
                        epochs = gr.Slider(1, 10, value=4, step=1, label="Epochs")
                    
                    train_btn = gr.Button("🚀 Start Training", variant="primary")
                    logs = gr.Textbox(label="", lines=15)
                    timer = gr.Timer(value=2, active=False)
                    
                    def start_train(data, ref, *args):
                        files = [item["path"] for item in data]
                        texts = [item["text"] for item in data]
                        threading.Thread(target=ui.train, args=(files, texts, ref, *args),daemon=True).start()
                        return gr.Timer(active=True), gr.Button(value="⏳ Training in Progress...", interactive=False)

                    train_btn.click(
                        start_train,
                        inputs=[data_state, ref_audio, model_name, speaker_name, base_model, lr, batch_size, epochs],
                        outputs=[timer, train_btn]
                    )
                    
                    def refresh():
                        s, l = ui.get_information()
                        if s=="Done":
                            return l, gr.Timer(active=False),gr.Button(value="🚀 Start Training", interactive=True)
                        else:
                            return l, gr.Timer(active= True),gr.Button(value="⏳ Training in Progress...", interactive=False)
                    
                    timer.tick(refresh, outputs=[logs, timer, train_btn])

                with gr.Tab("🎧 3 Test"):
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            label="Select Model",
                            choices=ui.get_models(),
                            value=None,
                            interactive=True,
                            scale=8,
                        )
                        refresh_models_btn = gr.Button("🔄 Refresh", scale=1)
                        refresh_models_btn.click( lambda: gr.update(choices=ui.get_models() ), outputs=[model_dropdown])
                    test_text = gr.Textbox(label="Text", value="你好，这是测试。", lines=3)
                    with gr.Row():
                        test_lang = gr.Dropdown(["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"], value="Chinese", label="Language")
                        test_speaker = gr.Textbox(label="Speaker", value="my_voice")
                    test_instruct = gr.Textbox(label="Instruction (Optional)", placeholder="e.g., 用温柔的语气说")
                    
                    test_btn = gr.Button("🎵 Generate", variant="primary")
                    audio_outputs = gr.State([])
                    
                    @gr.render(inputs=audio_outputs, triggers=[audio_outputs.change])
                    def render_audio_results(data):
                        if isinstance(data,tuple):
                            results, status = data
                            if not results:
                                gr.Markdown(status)
                                return
                            for sr, wav, name, checkpoint_path in results:
                                with gr.Row():
                                    gr.Audio(
                                        value=(sr, wav),
                                        label=f"🎵 {name}",
                                        autoplay=False,
                                        scale=4
                                    )
                                    with gr.Column(scale=1):
                                        save_name_input = gr.Textbox(
                                            label="保存名称",
                                            placeholder=f"{name}_custom",
                                            scale=1
                                        )
                                        save_btn = gr.Button("💾 保存", variant="secondary", scale=1)
                                        save_status = gr.Markdown("")
                                        
                                        save_btn.click(
                                            lambda: gr.update(interactive=False),
                                            outputs=[save_btn]
                                        ).then(
                                            lambda save_name: ui.save_model_checkpoint( checkpoint_path, save_name),
                                            inputs=[save_name_input],
                                            outputs=[save_status]
                                        )


                    def start_test(model, text, lang, speaker, instruct):
                        yield gr.State([]), gr.Button("⏳ Generating...", interactive=False)
                        yield ui.test_batch(model, text, lang, speaker, instruct),  gr.Button("🎵 Generate", variant="primary", interactive=True)
                    
                    test_btn.click(
                        start_test,
                        inputs=[model_dropdown, test_text, test_lang, test_speaker, test_instruct],
                        outputs=[audio_outputs, test_btn]
                    )
        
    return interface


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen3-TTS 语音合成与训练界面")
    parser.add_argument("-t", "--train", action="store_true", 
                        help="启动时默认打开训练标签页")
    parser.add_argument("-p", "--port", type=int, default=8886,
                        help="服务端口 (默认: 8886)")
    parser.add_argument("-s", "--server", type=str, default="0.0.0.0",
                        help="服务地址 (默认: 0.0.0.0)")
    parser.add_argument("-d", "--device", type=str, default=None,
                        help="设备 (例如: cpu, cuda:0, cuda:1)")
    parser.add_argument("-dt","--dtype", type=str, default=None,
                        choices=["bfloat16", "float16", "float32"],
                        help="数据类型 (默认: bfloat16)")    
    args = parser.parse_args()
    
    create_ui(
        train=args.train,
        device=args.device,
        dtype= args.dtype 
    ).launch(
        server_name=args.server, 
        server_port=args.port
    )