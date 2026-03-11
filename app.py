import os
import sys
import subprocess
import traceback
import gradio as gr
import numpy as np
import librosa
import spaces
import torch
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_URL = "https://github.com/fishaudio/fish-speech.git"
REPO_DIR = "fish-speech"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

os.chdir(REPO_DIR)
sys.path.insert(0, os.getcwd())

from fish_speech.models.text2semantic.inference import init_model, generate_long

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


def load_codec(codec_checkpoint_path, target_device, target_precision):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path("fish_speech/configs/modded_dac_vq.yaml"))
    codec = instantiate(cfg)

    state_dict = torch.load(codec_checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    codec.load_state_dict(state_dict, strict=False)
    codec.eval()
    codec.to(device=target_device, dtype=target_precision)
    return codec


codec_model = load_codec(os.path.join(checkpoint_dir, "codec.pth"), device, precision)


@torch.no_grad()
def encode_reference_audio(audio_path):
    wav_np, _ = librosa.load(audio_path, sr=codec_model.sample_rate, mono=True)
    wav = torch.from_numpy(wav_np).to(device)
    model_dtype = next(codec_model.parameters()).dtype
    audios = wav[None, None, :].to(dtype=model_dtype)
    audio_lengths = torch.tensor([wav.shape[0]], device=device, dtype=torch.long)
    indices, feature_lengths = codec_model.encode(audios, audio_lengths)
    return indices[0, :, : feature_lengths[0]]


@torch.no_grad()
def decode_codes_to_audio(merged_codes):
    audio = codec_model.from_indices(merged_codes[None])
    return audio[0, 0]


def estimate_duration(text):
    words = len(text.split())
    seconds = max(5, int(words * 0.4))
    return seconds


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
        if not text or not text.strip():
            raise gr.Error("Please enter some text to synthesize.")

        est = estimate_duration(text)
        gr.Info(f"Generating audio... estimated ~{est}s depending on text length.")

        prompt_tokens_list = None
        if ref_audio is not None and ref_text and ref_text.strip():
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

        merged_codes = codes[0] if len(codes) == 1 else torch.cat(codes, dim=1)
        merged_codes = merged_codes.to(device)

        audio_waveform = decode_codes_to_audio(merged_codes)
        audio_np = audio_waveform.cpu().float().numpy()
        audio_np = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (codec_model.sample_rate, audio_np)

    except gr.Error:
        raise
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Inference error: {str(e)}")


TAGS = [
    "[pause]", "[emphasis]", "[laughing]", "[inhale]", "[chuckle]", "[tsk]",
    "[singing]", "[excited]", "[laughing tone]", "[interrupting]", "[chuckling]",
    "[excited tone]", "[volume up]", "[echo]", "[angry]", "[low volume]", "[sigh]",
    "[low voice]", "[whisper]", "[screaming]", "[shouting]", "[loud]", "[surprised]",
    "[short pause]", "[exhale]", "[delight]", "[panting]", "[audience laughter]",
    "[with strong accent]", "[volume down]", "[clearing throat]", "[sad]",
    "[moaning]", "[shocked]",
]

TAGS_HTML = " ".join(
    f'<code style="margin:2px;display:inline-block">{t}</code>' for t in TAGS
)

with gr.Blocks(title="Fish Audio S2 Pro") as app:

    gr.Markdown(
        f"""
        <div style="text-align:center;max-width:900px;margin:0 auto;padding:24px 0 8px">
            <h1 style="font-size:2.4rem;font-weight:800;color:#1E3A8A;margin-bottom:6px">
                🐟 Fish Audio S2 Pro
            </h1>
            <p style="font-size:1.05rem;color:#4B5563;margin-bottom:8px">
                State-of-the-Art Dual-Autoregressive Text-to-Speech &nbsp;·&nbsp;
                <a href="https://huggingface.co/fishaudio/s2-pro" target="_blank" style="color:#2563EB">Model Page ↗</a>
                &nbsp;·&nbsp;
                <a href="https://github.com/fishaudio/fish-speech" target="_blank" style="color:#2563EB">GitHub ↗</a>
            </p>
            <p style="font-size:0.95rem;color:#6B7280">
                80+ languages supported · Zero-shot voice cloning · 15,000+ inline emotion tags
            </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### ✍️ Input Text")
            text_input = gr.Textbox(
                show_label=False,
                placeholder="Type the text you want to synthesize.\nLanguage is auto-detected — write in any language.\nAdd emotion tags like [laugh] or [whisper in small voice] anywhere in the text.",
                lines=7,
            )

            with gr.Accordion("🎙️ Voice Cloning — Optional", open=False):
                gr.Markdown(
                    "Upload a clean **5–10 second** audio clip and provide its exact transcription. "
                    "The model will clone that voice for synthesis. Language is inferred automatically."
                )
                ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                ref_text = gr.Textbox(
                    label="Reference Audio Transcription",
                    placeholder="Exact transcription of the reference audio...",
                )

            with gr.Accordion("⚙️ Advanced Settings", open=False):
                with gr.Row():
                    max_new_tokens = gr.Slider(0, 2048, 1024, step=8, label="Max New Tokens (0 = auto)")
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
                f"""
                <div style="background:#EFF6FF;padding:16px;border-radius:10px;margin-top:16px">
                    <h4 style="margin:0 0 8px;color:#1D4ED8">🏷️ Supported Emotion Tags</h4>
                    <p style="font-size:0.85rem;color:#374151;margin-bottom:8px">
                        15,000+ unique tags supported. Use free-form descriptions like
                        <code>[whisper in small voice]</code> or <code>[professional broadcast tone]</code>.
                        Common tags:
                    </p>
                    <div style="line-height:2">{TAGS_HTML}</div>
                </div>
                """
            )

    gr.Markdown(
        """
        <div style="background:#F0FDF4;padding:16px;border-radius:10px;margin-top:8px">
            <h4 style="margin:0 0 8px;color:#166534">🌍 Supported Languages</h4>
            <p style="font-size:0.9rem;color:#374151;margin:0">
                <strong>Tier 1:</strong> Japanese · English · Chinese &nbsp;|&nbsp;
                <strong>Tier 2:</strong> Korean · Spanish · Portuguese · Arabic · Russian · French · German<br>
                <strong>Also supported:</strong> sv, it, tr, no, nl, cy, eu, ca, da, gl, ta, hu, fi, pl, et, hi,
                la, ur, th, vi, jw, bn, yo, sl, cs, sw, nn, he, ms, uk, id, kk, bg, lv, my, tl, sk, ne, fa,
                af, el, bo, hr, ro, sn, mi, yi, am, be, km, is, az, sd, br, sq, ps, mn, ht, ml, sr, sa, te,
                ka, bs, pa, lt, kn, si, hy, mr, as, gu, fo, and more.
                Language is <strong>auto-detected</strong> from the input text — no configuration needed.
            </p>
        </div>
        """
    )

    gr.Markdown("### 🌟 Examples")
    gr.Examples(
        examples=[
            ["Hello world! This is a test of the Fish Audio S2 Pro model.", None, "", 1024, 200, 0.7, 1.2, 0.7],
            ["I can't believe it! [laugh] This is absolutely amazing!", None, "", 1024, 200, 0.7, 1.2, 0.7],
            ["[whisper in small voice] I have a secret to tell you... promise you won't tell anyone?", None, "", 1024, 200, 0.7, 1.2, 0.7],
            ["Olá! Este modelo suporta português nativamente, sem configuração extra.", None, "", 1024, 200, 0.7, 1.2, 0.7],
            ["[excited] 日本語も話せます！すごいでしょう？", None, "", 1024, 200, 0.7, 1.2, 0.7],
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