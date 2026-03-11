import os
import sys
import subprocess
import traceback
import gradio as gr
import numpy as np
import librosa
import spaces
import torch
from huggingface_hub import snapshot_download

REPO_URL = "https://github.com/fishaudio/fish-speech.git"
REPO_DIR = "fish-speech"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, os.getcwd())

from fish_speech.models.text2semantic.inference import (
    init_model,
    generate_long,
    load_codec_model,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
precision = torch.bfloat16

checkpoint_dir = snapshot_download(repo_id="fishaudio/s2-pro")

llama_model, decode_one_token = init_model(
    checkpoint_path=checkpoint_dir,
    device=device,
    precision=precision,
    compile=False,
)

with torch.device(device):
    llama_model.setup_caches(
        max_batch_size=1,
        max_seq_len=llama_model.config.max_seq_len,
        dtype=next(llama_model.parameters()).dtype,
    )

codec_checkpoint = os.path.join(checkpoint_dir, "codec.pth")
codec_model = load_codec_model(codec_checkpoint, device=device, precision=precision)


@torch.no_grad()
def encode_reference_audio(audio_path):
    wav_np, _ = librosa.load(audio_path, sr=codec_model.sample_rate, mono=True)
    wav = torch.from_numpy(wav_np).to(device)
    model_dtype = next(codec_model.parameters()).dtype
    audios = wav[None, None, :].to(dtype=model_dtype)
    audio_lengths = torch.tensor([wav.shape[0]], device=device, dtype=torch.long)
    indices, feature_lengths = codec_model.encode(audios, audio_lengths)
    return indices[0, :, : feature_lengths[0]]


def decode_codes_to_audio(merged_codes):
    with torch.inference_mode(False):
        with torch.no_grad():
            codes_clean = merged_codes.clone()
            audio = codec_model.from_indices(codes_clean[None])
            return audio[0, 0]


@spaces.GPU(duration=120)
def tts_inference(
    text,
    ref_audio,
    ref_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
):
    try:
        prompt_tokens_list = None

        if ref_audio is not None and ref_text:
            prompt_tokens_list = [encode_reference_audio(ref_audio).cpu()]

        generator = generate_long(
            model=llama_model,
            device=device,
            decode_one_token=decode_one_token,
            text=text,
            num_samples=1,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=30,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            compile=False,
            iterative_prompt=True,
            chunk_length=chunk_length,
            prompt_text=[ref_text] if ref_text else None,
            prompt_tokens=prompt_tokens_list,
        )

        codes = []
        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)
            elif response.action == "next":
                break

        if not codes:
            raise gr.Error("No audio was generated. Please check your input text.")

        merged_codes = torch.cat(codes, dim=1).to(device)
        audio_waveform = decode_codes_to_audio(merged_codes)
        audio_np = audio_waveform.cpu().float().numpy()

        return (codec_model.sample_rate, audio_np)

    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Inference error: {str(e)}")


custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(title="Fish Audio S2 Pro") as app:

    gr.Markdown(
        """
        <div style="text-align: center; max-width: 800px; margin: 0 auto; padding: 20px 0;">
            <h1 style="font-size: 2.5rem; font-weight: 800; color: #1E3A8A; margin-bottom: 10px;">
                🐟 Fish Audio S2 Pro
            </h1>
            <p style="font-size: 1.1rem; color: #4B5563;">
                State-of-the-Art Dual-Autoregressive Text-to-Speech.<br>
                Supports 80+ languages, emotion tags (e.g. <code>[laugh]</code>, <code>[whisper]</code>) and zero-shot voice cloning.
            </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### ✍️ Input Text")
            text_input = gr.Textbox(
                show_label=False,
                placeholder="Type the text you want to synthesize here.\nTry adding tags like [laugh], [whisper], or [angry]!",
                lines=7,
            )

            with gr.Accordion("🎙️ Voice Cloning (Optional Reference)", open=False):
                gr.Markdown("Upload a clean 5–10 second audio clip and type exactly what is said in it to clone the voice.")
                ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                ref_text = gr.Textbox(
                    label="Reference Audio Transcription",
                    placeholder="Exact transcription of the reference audio...",
                )

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    max_new_tokens = gr.Slider(0, 2048, 1024, step=8, label="Max New Tokens (0 = no limit)")
                    chunk_length = gr.Slider(100, 400, 200, step=8, label="Chunk Length")
                with gr.Row():
                    top_p = gr.Slider(0.1, 1.0, 0.7, step=0.01, label="Top-P")
                    repetition_penalty = gr.Slider(0.9, 2.0, 1.2, step=0.01, label="Repetition Penalty")
                    temperature = gr.Slider(0.1, 1.0, 0.7, step=0.01, label="Temperature")

            generate_btn = gr.Button("🚀 Generate Audio", variant="primary", size="lg")

        with gr.Column(scale=4):
            gr.Markdown("### 🎧 Result")
            audio_output = gr.Audio(
                label="Generated Audio",
                type="numpy",
                interactive=False,
                autoplay=True,
            )

            gr.Markdown(
                """
                <div style="background-color: #EFF6FF; padding: 15px; border-radius: 8px; margin-top: 20px;">
                    <h4 style="margin-top: 0; color: #1D4ED8;">💡 Pro Tips</h4>
                    <ul style="margin-bottom: 0; color: #1E3A8A; font-size: 0.95rem;">
                        <li>The model understands natural text — no need for manual phonemes.</li>
                        <li>Control emotion with brackets: <i>[pitch up] Wow! [laugh]</i></li>
                        <li>For cloning, the more accurate the transcription, the better the result.</li>
                    </ul>
                </div>
                """
            )

    gr.Markdown("### 🌟 Examples")
    gr.Examples(
        examples=[
            ["Hello world! This is a test of the Fish Audio S2 Pro model.", None, "", 1024, 200, 0.7, 1.2, 0.7],
            ["I can't believe it! [laugh] This is absolutely amazing!", None, "", 1024, 200, 0.7, 1.2, 0.7],
            ["[whisper in small voice] I have a secret to tell you... promise you won't tell anyone?", None, "", 1024, 200, 0.7, 1.2, 0.7],
        ],
        inputs=[text_input, ref_audio, ref_text, max_new_tokens, chunk_length, top_p, repetition_penalty, temperature],
        outputs=[audio_output],
        fn=tts_inference,
        cache_examples=False,
    )

    generate_btn.click(
        fn=tts_inference,
        inputs=[text_input, ref_audio, ref_text, max_new_tokens, chunk_length, top_p, repetition_penalty, temperature],
        outputs=[audio_output],
    )

if __name__ == "__main__":
    app.launch()