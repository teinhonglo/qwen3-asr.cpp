# Qwen3-ASR.cpp

A high-performance C++ implementation of Qwen3-SLU using the GGML tensor library. Optimized for Apple Silicon with Metal GPU acceleration, providing fast speech recognition and word-level timestamp alignment.

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

# Build ggml
cd ggml
mkdir build && cd build
cmake ..
cmake --build . --config Release -j 8
cd ../../

# Build qwen3-asr.cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j8
cd ../
```
## Download and Unpack the Data and Model

```bash
tar -zxf demos_and_models.tar.gz
cp demos/path.sh .
```

## Quick Start

### 1. Transcription (SLU)

Transcribe audio files to text:

#### Example 1: Multi-intent SLU output

```bash
# Add context instruction to bias decoding
. ./path.sh

./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-asr-slu-f16.gguf -f demos/id_26.wav -c "$prompt"

# Output Formats
# language None{"asr_text": "打开座椅通风打开座椅按摩", "semantics": "[{\"domain\": \"车载控制\", \"intent\": \"车身控制\", \"slots\": {\"操作\": \"打开\", \"对象\": \"座椅\", \"对象功能\": \"通风\"}}, {\"domain\": \"车载控制\", \"intent\": \"车身控制\", \"slots\": {\"操作\": \"打开\", \"对象\": \"座椅\", \"对象功能\": \"按摩\"}}]"}
```

#### Example 2: No matched intent

```bash
# Add context instruction to bias decoding
. ./path.sh

./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-asr-slu-f16.gguf -f demos/id_19152.wav -c "$prompt"

# Output Formats
# language None{"asr_text": "离离原上草这首诗你会背吗", "semantics": "[]"}
```

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
cmake --build . -j 8
cd ../
. ./path.sh

# Run with --profile flag
./build/qwen3-asr-cli -m models/qwen3-asr-0.6b-asr-slu-f16.gguf -f demos/id_26.wav -c "$prompt" --profile
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
