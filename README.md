# Qwen3-ASR.cpp

A high-performance C++ implementation of Qwen3-ASR and Qwen3-ForcedAligner using the GGML tensor library. Optimized for Apple Silicon with Metal GPU acceleration, providing fast speech recognition and word-level timestamp alignment.

## Features

- **Automatic Speech Recognition (ASR)**: Transcribe audio files to text in 30+ languages
- **Flash Attention**: Uses `ggml_flash_attn_ext()` for fast decoding (3.7x speedup)
- **Metal GPU Acceleration**: Optimized for Apple Silicon with dual CPU+Metal backend
- **Accelerate/vDSP**: Highly optimized mel spectrogram computation (45x speedup)
- **mmap Weight Loading**: Zero-copy GPU transfer for fast model initialization
- **F16 KV Cache**: Reduced memory bandwidth with half-precision key-value cache
- **Quantization Support**: Q8_0 quantization for reduced memory usage (~40% smaller)
- **Pure C++17**: No Python runtime required for inference

## Supported Models

| Model | Size | Description |
|-------|------|-------------|
| `qwen3-asr-0.6b-asr-slu-f16.gguf` | ~1.8 GB | ASR-SLU model, F16 precision |
| `qwen3-asr-0.6b-asr-slu-q8_0.gguf` | ~1.3 GB | ASR-SLU model, Q8_0 quantized |

## Requirements

- CMake 3.14+
- C++17 compatible compiler (Clang 7+, GCC 8+, MSVC 2019+)
- Apple Silicon recommended (Metal GPU support)
- GGML library (included as submodule)

## Building

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/teinhonglo/qwen3-asr.cpp.git
cd qwen3-asr.cpp

# Build
cd ggml
mkdir build && cd build
cmake ..
cmake --build . --config Release -j 8
cd ../../

# Build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
cd ../
```
## Download and Unpack the Data and Model

```bash
tar -zxf demos.tar.gz
tar -zxf models.tar.gz
```

## Quick Start

### 1. Transcription (SLU)

Transcribe audio files to text:

```bash
# Add context instruction to bias decoding
prompt="你是一个专业的车载系统自然语言理解（NLU）专家。\n你的任务是基于用户的查询（Query），同时完成两项任务：\n1.  意图识别 (Intent Classification): 识别出查询中包含的所有领域（Domain）和意图（Intent）。\n2.  槽位填充 (Slot Filling): 抽取出与每个意图相关的槽位（Slot）和槽位值（Value）。\n\n你需要严格遵循以下规则：\n1.  识别多个语义帧: 用户的单次查询可能包含多个独立的意图。你需要为每一个意图生成一个对应的语义结构。\n2.  输出格式: 你的输出必须是一个严格的 JSON List (列表)。\n3.  列表中的每一个 JSON 对象都必须包含且只包含这三个欄位：\"domain\"、\"intent\"、\"slots\"。\n4.  \"slots\" 必须是 JSON object；若该意图无槽位，請輸出空物件 {}。\n5.  如果没有匹配到任何领域和意图，请返回空列表 []。\n6.  最终回答中除了 JSON，不要包含其他文字。\n\n输出格式范例：\n- 单一语义帧：\n[{\"domain\":\"地图\",\"intent\":\"导航\",\"slots\":{\"终点目标\":\"广州塔\"}}]\n\n- 多语义帧：\n[\n  {\"domain\":\"地图\",\"intent\":\"导航\",\"slots\":{\"终点目标\":\"公司\"}},\n  {\"domain\":\"音乐\",\"intent\":\"播放音乐\",\"slots\":{\"歌曲名\":\"夜曲\"}}\n]\n\n- 无槽位：\n[{\"domain\":\"播放控制\",\"intent\":\"播放控制\",\"slots\":{}}]\n\n- 无匹配：\n[]\n"

./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-asr-slu-f16.gguf -f demos/id_26.wav -c "$prompt"

# Output Formats
language None打开座椅通风打开座椅按摩<slu>[{"domain": "车载控制", "intent": "车身控制", "slots": {"操作": "打开", "对象": "座椅", "对象功能": "通风"}}, {"domain": "车载控制", "intent": "车身控制", "slots": {"操作": "打开", "对象": "座椅", "对象功能": "按摩"}}]

./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-asr-slu-f16.gguf -f demos/id_19252.wav -c "$prompt"

# Output Formats
language None离离原上草这首诗你会背吗<slu>[]
```

### Key Optimizations

- **Flash Attention** (`ggml_flash_attn_ext`): 3.7x decode speedup vs. standard attention
- **Metal GPU Dual Backend**: Automatic scheduling between CPU and GPU for optimal performance
- **mmap + Zero-Copy GPU Transfer**: Fast model loading via `ggml_backend_dev_buffer_from_host_ptr`
- **F16 KV Cache**: Half-precision key-value cache reduces memory bandwidth
- **Selective Logits**: Only compute last token logits for lm_head (saves computation)
- **Weight Tying**: token_embd = output weight (saves memory)
- **vDSP/Accelerate Mel**: 45x speedup for mel spectrogram computation on Apple platforms

## Audio Requirements

- **Format**: WAV (PCM)
- **Sample rate**: 16 kHz
- **Channels**: Mono
- **Bit depth**: 16-bit

Convert audio with ffmpeg:
```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

## Performance Profiling

Build with timing instrumentation to see detailed breakdowns:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DQWEN3_ASR_TIMING=ON
cmake --build . -j$(sysctl -n hw.ncpu)

# Run with --profile flag
./qwen3-asr-cli -m models/qwen3-asr-0.6b-f16.gguf -f sample.wav --profile
```

For production builds, omit `-DQWEN3_ASR_TIMING=ON` to remove timing overhead.

## Project Structure

```
qwen3-asr.cpp/
├── src/
│   ├── main.cpp              # CLI entry point
│   ├── qwen3_asr.cpp/h       # High-level ASR API
│   ├── forced_aligner.cpp/h  # Forced alignment implementation
│   ├── audio_encoder.cpp/h   # Audio feature encoder
│   ├── text_decoder.cpp/h    # Text decoder (Qwen2 architecture)
│   ├── mel_spectrogram.cpp/h # Mel spectrogram computation
│   ├── audio_injection.cpp/h # Audio-text embedding injection
│   ├── gguf_loader.cpp/h     # GGUF model loading
│   └── timing.h              # Timing instrumentation macros
├── tests/
│   ├── test_mel.cpp          # Mel spectrogram tests
│   ├── test_encoder.cpp      # Audio encoder tests
│   ├── test_decoder.cpp      # Text decoder tests
│   └── reference/            # Reference data for validation
├── scripts/
│   └── convert_hf_to_gguf.py # Model conversion script
├── assets/
│   └── korean_dict_jieba.dict # Korean word dictionary (17,968 words)
├── models/                   # GGUF model files (not tracked in git)
├── ggml/                     # GGML library (git submodule)
└── CMakeLists.txt
```

## License

This project is licensed under the MIT License. See LICENSE for details.

## Acknowledgments

- [GGML](https://github.com/ggerganov/ggml) - Tensor library for machine learning
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) - Original model by Alibaba
- [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) - Original aligner model by Alibaba

---
