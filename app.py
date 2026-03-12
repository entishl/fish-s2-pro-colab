import os
import sys
import subprocess
import traceback
import gradio as gr
import numpy as np
import librosa
import torch
from pathlib import Path
from huggingface_hub import snapshot_download

# --- Configuration ---
REPO_URL = "https://github.com/fishaudio/fish-speech.git"
REPO_DIR = "fish-speech"
MODEL_REPO_ID = "fishaudio/s2-pro"

# --- Environment Setup ---
if not os.path.exists(REPO_DIR):
    print(f"Cloning {REPO_URL}...")
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

# Ensure the cloned repo is in path
repo_path = os.path.abspath(REPO_DIR)
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

# Import model logic after sys.path is set
from fish_speech.models.text2semantic.inference import init_model, generate_long

device = "cuda" if torch.cuda.is_available() else "cpu"
precision = torch.bfloat16 if device == "cuda" else torch.float32

# --- Model Loading ---
print(f"Downloading model from {MODEL_REPO_ID} (using HF_TOKEN if available)...")
hf_token = os.getenv("HF_TOKEN")
checkpoint_dir = snapshot_download(
    repo_id=MODEL_REPO_ID, 
    token=hf_token
)

print(f"Initializing Llama model on {device}...")
llama_model, decode_one_token = init_model(
    checkpoint_path=checkpoint_dir,
    device=device,
    precision=precision,
    compile=False,
)

if device == "cuda":
    with torch.device(device):
        llama_model.setup_caches(
            max_batch_size=1,
            max_seq_len=llama_model.config.max_seq_len,
            dtype=next(llama_model.parameters()).dtype,
        )

def load_codec(codec_checkpoint_path, target_device, target_precision):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    # Path adjustment for Colab structure
    cfg_path = Path(REPO_DIR) / "fish_speech/configs/modded_dac_vq.yaml"
    if not cfg_path.exists():
        cfg_path = Path("fish_speech/configs/modded_dac_vq.yaml")
        
    cfg = OmegaConf.load(cfg_path)
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

print("Loading Codec model...")
codec_model = load_codec(os.path.join(checkpoint_dir, "codec.pth"), device, precision)

# --- Inference Functions ---

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

whisper_model = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        from faster_whisper import WhisperModel
        print("Loading Whisper large-v3...")
        # Use float16 for T4 GPU efficiency
        compute_type = "float16" if device == "cuda" else "int8"
        whisper_model = WhisperModel("large-v3", device=device, compute_type=compute_type)
    return whisper_model

def transcribe_audio(audio_path):
    if audio_path is None:
        raise gr.Error("Please upload a reference audio file first.")
    try:
        gr.Info("Transcribing audio...")
        model = get_whisper_model()
        segments, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        if not text:
            raise gr.Error("Whisper could not detect any speech.")
        return text
    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Transcription error: {str(e)}")

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
            raise gr.Error("Please enter some text.")

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
            raise gr.Error("No audio was generated.")

        merged_codes = codes[0] if len(codes) == 1 else torch.cat(codes, dim=1)
        merged_codes = merged_codes.to(device)

        audio_waveform = decode_codes_to_audio(merged_codes)
        audio_np = audio_waveform.cpu().float().numpy()
        audio_np = (audio_np * 32767).clip(-32768, 32767).astype(np.int16)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (codec_model.sample_rate, audio_np)

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Inference error: {str(e)}")

# --- UI Layout ---

TAGS = [
    "[pause]", "[emphasis]", "[laughing]", "[inhale]", "[chuckle]", "[tsk]",
    "[singing]", "[excited]", "[laughing tone]", "[interrupting]", "[chuckling]",
    "[excited tone]", "[volume up]", "[echo]", "[angry]", "[low volume]", "[sigh]",
    "[low voice]", "[whisper]", "[screaming]", "[shouting]", "[loud]", "[surprised]",
    "[short pause]", "[exhale]", "[delight]", "[panting]", "[audience laughter]",
    "[with strong accent]", "[volume down]", "[clearing throat]", "[sad]",
    "[moaning]", "[shocked]",
]
TAGS_HTML = " ".join(f'<code style="margin:2px;display:inline-block">{t}</code>' for t in TAGS)

with gr.Blocks(title="Fish Audio S2 Pro - Colab") as web_app:
    gr.Markdown("# 🐟 Fish Audio S2 Pro (Colab Optimized)")
    
    with gr.Row():
        with gr.Column(scale=5):
            text_input = gr.Textbox(label="Input Text", lines=7, placeholder="Type here...")
            
            with gr.Accordion("🎙️ Voice Cloning", open=False):
                ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                transcribe_btn = gr.Button("🎤 Auto-transcribe", variant="secondary")
                ref_text = gr.Textbox(label="Transcription")

            with gr.Accordion("⚙️ Settings", open=False):
                max_new_tokens = gr.Slider(0, 2048, 1024, step=8, label="Max Tokens")
                chunk_length = gr.Slider(100, 400, 200, step=8, label="Chunk Length")
                with gr.Row():
                    top_p = gr.Slider(0.1, 1.0, 0.7, label="Top-P")
                    rep_penalty = gr.Slider(0.9, 2.0, 1.2, label="Repetition Penalty")
                    temp = gr.Slider(0.1, 1.0, 0.7, label="Temperature")

            generate_btn = gr.Button("🚀 Generate", variant="primary")

        with gr.Column(scale=4):
            audio_output = gr.Audio(label="Result", autoplay=True)
            gr.Markdown(f"### 🏷️ Emotion Tags\n{TAGS_HTML}")

    transcribe_btn.click(transcribe_audio, [ref_audio], [ref_text])
    generate_btn.click(
        tts_inference, 
        [text_input, ref_audio, ref_text, max_new_tokens, chunk_length, top_p, rep_penalty, temp], 
        [audio_output]
    )

if __name__ == "__main__":
    web_app.launch(share=True, debug=True, show_error=True)