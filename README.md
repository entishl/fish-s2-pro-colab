---
title: Fish Audio S2 Pro
emoji: 🐟
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
pinned: false
license: other
license_name: fish-audio-research-license
license_link: LICENSE
short_description: Zero GPU Text-to-Speech using Fish Audio S2 Pro
---

# Fish Audio S2 Pro - Zero GPU Space

This is a beautiful, self-contained Gradio interface for [Fish Audio S2 Pro](https://huggingface.co/fishaudio/s2-pro) designed to run on Hugging Face's **Zero GPU** architecture.

It utilizes a Dual-Autoregressive (Dual-AR) architecture to achieve highly expressive, natural text-to-speech with fine-grained control via natural language tags (e.g., `[laugh]`, `[whisper]`) and zero-shot voice cloning capabilities.

## Features
- **Zero GPU Compatibility:** Optimized for dynamic GPU allocation via the `@spaces.GPU` decorator.
- **Voice Cloning:** Simply upload a 5-10s reference audio and its transcript to clone a voice instantly.
- **Emotion & Prosody Control:** Inject tags directly into your text.

## License
Released under the **FISH AUDIO RESEARCH LICENSE**.