# Qwen3-TTS Speech Synthesis and Training System
[中文版本](README_CN.md) | English
## Overview

This is a speech synthesis and training system based on the Qwen3-TTS model, providing complete model training, testing, and inference capabilities. It supports custom voice cloning, multi-speaker dialogue generation, and a visual training management interface.

## Main Features

### 1. Speech Generation (Generate)
- **Single Text Generation**: Supports speech synthesis for single text segments
- **Multi-line Text Generation**: Supports sentence-by-sentence editing and batch processing
- **Emotion Control**: Supports emotional instruction control for speech expression
- **Real-time Preview**: Real-time viewing and audio playback during generation

### 2. Multi-speaker Dialogue Generation (Generates)
- **Format Parsing**: Supports text parsing in `speaker[emotion]:content` format
- **Dynamic Editing**: Individual adjustment of speaker, language, and emotion for each sentence
- **Batch Generation**: One-click generation of all dialogue audio
- **Audio Preview**: Independent preview and download for each sentence

### 3. Model Training (Train)
- **Data Preparation**: Supports audio file upload and automatic speech recognition
- **Training Configuration**: Adjustable learning rate, batch size, training epochs, and other parameters
- **Real-time Monitoring**: Real-time display of logs and status during training
- **Checkpoint Management**: Support for testing and saving multiple training checkpoints
## Screenshots

### Training Workflow
![Training Preparation](train_prepare.png)
*Data preparation and training configuration interface*

![Training Process](training.png)
*Real-time training progress monitoring*

![Training Testing](train_test.png)
*Model testing and checkpoint evaluation*

### Speech Generation
![Single Speaker](single_speaker.png)
*Single text speech generation interface*

![Multi Speaker](multy_speaker.png)
*Multi-speaker dialogue generation interface*

## Installation and Usage

### Environment Requirements
```bash
pip install qwen-tts
pip install funasr  # For speech recognition
```

## Model Preparation

Models will be automatically downloaded on first use, or you can manually download them:

```bash
modelscope download --model Qwen/Qwen3-TTS-Tokenizer-12Hz  --local_dir ./Qwen3-TTS-Tokenizer-12Hz 
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir ./Qwen3-TTS-12Hz-1.7B-Base
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --local_dir ./Qwen3-TTS-12Hz-0.6B-Base
```
```
|-- Qwen3-TTS-12Hz-1.7B-Base
|-- Qwen3-TTS-12Hz-0.6B-Base
|-- Qwen3-TTS-Tokenizer-12Hz
├── multy_speaker.png
├── README_CN.md
├── README.md
├── single_speaker.png
├── train_prepare.png
├── train_test.png
├── training.png
├── generate.py
└── main.py
```
### Startup Methods

#### 1. Basic Usage (Generation Only)
```bash
python main.py
```

#### 2. Full Features (Including Training)
```bash
python main.py --train
```

#### 3. Custom Configuration
```bash
python main.py --train --port 8080 --device cuda:0 --dtype bfloat16
```

### Parameter Description
- `--train` / `-t`: Enable training functionality
- `--port` / `-p`: Service port (default: 8886)
- `--server` / `-s`: Service address (default: 0.0.0.0)
- `--device` / `-d`: Computing device (cuda:0, cpu, etc.)
- `--dtype`: Data precision (bfloat16, float16, float32)

