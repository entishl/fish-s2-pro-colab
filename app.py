import os
import sys
import subprocess
import traceback
import gradio as gr
import numpy as np
import spaces
import torch
from huggingface_hub import snapshot_download

REPO_URL = "https://github.com/fishaudio/fish-speech.git"
REPO_DIR = "fish-speech"

if not os.path.exists(REPO_DIR):
    print(f"Clonando o repositório de {REPO_URL}...")
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    print("Repositório clonado com sucesso!")

os.chdir(REPO_DIR)
sys.path.insert(0, os.getcwd())

from fish_speech.models.text2semantic.inference import (
    init_model,
    generate_long,
    load_codec_model,
    decode_to_audio,
    encode_audio
)

device = "cuda" if torch.cuda.is_available() else "cpu"
precision = torch.bfloat16

print("Baixando os pesos do Fish Audio S2 Pro...")
checkpoint_dir = snapshot_download(repo_id="fishaudio/s2-pro")

print("Carregando o modelo LLAMA (isso pode levar alguns instantes)...")
llama_model, decode_one_token = init_model(
    checkpoint_path=checkpoint_dir, 
    device=device, 
    precision=precision, 
    compile=False
)

with torch.device(device):
    llama_model.setup_caches(
        max_batch_size=1,
        max_seq_len=llama_model.config.max_seq_len,
        dtype=next(llama_model.parameters()).dtype,
    )

print("Carregando o modelo Codec (VQGAN)...")
codec_checkpoint = os.path.join(checkpoint_dir, "codec.pth")
codec_model = load_codec_model(codec_checkpoint, device=device, precision=precision)

print("✅ Todos os modelos carregados com sucesso!")

@spaces.GPU(duration=120)
def tts_inference(
    text, 
    ref_audio, 
    ref_text, 
    max_new_tokens, 
    chunk_length, 
    top_p, 
    repetition_penalty, 
    temperature
):
    try:
        prompt_tokens_list = None
        
        if ref_audio is not None and ref_text:
            prompt_tokens_list = [encode_audio(ref_audio, codec_model, device).cpu()]
            
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
            raise gr.Error("Nenhum áudio foi gerado. Verifique o seu texto de entrada.")
            
        merged_codes = torch.cat(codes, dim=1)
        audio_waveform = decode_to_audio(merged_codes.to(device), codec_model)
        audio_np = audio_waveform.cpu().float().numpy()
        
        return (codec_model.sample_rate, audio_np)

    except Exception as e:
        traceback.print_exc()
        raise gr.Error(f"Erro na Inferência: {str(e)}")


custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    block_title_text_weight="600",
    block_border_width="1px",
    block_shadow="0px 2px 4px rgba(0, 0, 0, 0.05)",
    button_shadow="0px 2px 4px rgba(0, 0, 0, 0.1)",
)

with gr.Blocks(theme=custom_theme, title="Fish Audio S2 Pro") as app:
    
    gr.Markdown(
        """
        <div style="text-align: center; max-width: 800px; margin: 0 auto; padding: 20px 0;">
            <h1 style="font-size: 2.5rem; font-weight: 800; color: #1E3A8A; margin-bottom: 10px;">
                🐟 Fish Audio S2 Pro
            </h1>
            <p style="font-size: 1.1rem; color: #4B5563;">
                State-of-the-Art Dual-Autoregressive Text-to-Speech.<br>
                Suporta mais de 80 idiomas, controle emocional no texto (ex: <code>[laugh]</code>, <code>[whisper]</code>) e clonagem de voz Zero-Shot.
            </p>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### ✍️ Texto de Entrada")
            text_input = gr.Textbox(
                show_label=False,
                placeholder="Digite o texto que você deseja sintetizar aqui.\nTente adicionar tags como [laugh], [whisper], ou [angry]!", 
                lines=7
            )
            
            with gr.Accordion("🎙️ Clonagem de Voz (Referência Opcional)", open=False):
                gr.Markdown("Faça upload de um áudio limpo de 5 a 10 segundos e digite exatamente o que é dito nele para clonar a voz.")
                ref_audio = gr.Audio(label="Áudio de Referência", type="filepath")
                ref_text = gr.Textbox(label="Texto do Áudio", placeholder="Transcrição exata do áudio de referência...")
            
            with gr.Accordion("⚙️ Configurações Avançadas", open=False):
                with gr.Row():
                    max_new_tokens = gr.Slider(0, 2048, 1024, step=8, label="Max New Tokens (0 = sem limite)")
                    chunk_length = gr.Slider(100, 400, 200, step=8, label="Tamanho do Chunk")
                with gr.Row():
                    top_p = gr.Slider(0.1, 1.0, 0.7, step=0.01, label="Top-P")
                    repetition_penalty = gr.Slider(0.9, 2.0, 1.2, step=0.01, label="Penalidade de Repetição")
                    temperature = gr.Slider(0.1, 1.0, 0.7, step=0.01, label="Temperatura")
                
            generate_btn = gr.Button("🚀 Gerar Áudio", variant="primary", size="lg")
            
        with gr.Column(scale=4):
            gr.Markdown("### 🎧 Resultado")
            audio_output = gr.Audio(label="Áudio Gerado", type="numpy", interactive=False, autoplay=True)
            
            gr.Markdown(
                """
                <div style="background-color: #EFF6FF; padding: 15px; border-radius: 8px; margin-top: 20px;">
                    <h4 style="margin-top: 0; color: #1D4ED8;">💡 Dicas Profissionais</h4>
                    <ul style="margin-bottom: 0; color: #1E3A8A; font-size: 0.95rem;">
                        <li>O modelo compreende texto natural perfeitamente, sem necessidade de fonemas manuais.</li>
                        <li>Envolva palavras com colchetes para ditar emoções. Ex: <i>[pitch up] Uau! [laugh]</i>.</li>
                        <li>Para clonagem, quanto mais exata a transcrição do áudio de base, melhor o resultado.</li>
                    </ul>
                </div>
                """
            )
            
    gr.Markdown("### 🌟 Exemplos")
    gr.Examples(
        examples=[
            ["Hello world! This is a test of the Fish Audio S2 Pro model.", None, "", 1024, 200, 0.7, 1.2, 0.7],
            ["I can't believe it! [laugh] This is absolutely amazing!", None, "", 1024, 200, 0.7, 1.2, 0.7],
            ["[whisper in small voice] I have a secret to tell you... promise you won't tell anyone?", None, "", 1024, 200, 0.7, 1.2, 0.7]
        ],
        inputs=[text_input, ref_audio, ref_text, max_new_tokens, chunk_length, top_p, repetition_penalty, temperature],
        outputs=[audio_output],
        fn=tts_inference,
        cache_examples=False,
    )
    
    generate_btn.click(
        fn=tts_inference,
        inputs=[text_input, ref_audio, ref_text, max_new_tokens, chunk_length, top_p, repetition_penalty, temperature],
        outputs=[audio_output]
    )

if __name__ == "__main__":
    app.launch()