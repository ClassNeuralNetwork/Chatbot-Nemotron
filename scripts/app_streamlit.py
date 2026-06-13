import os
import gc
import time
import torch
from pathlib import Path
import streamlit as st
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# O uso dessa variavel de ambiente foi porque o bitsandbytes não suporta a versão
# mais recente do cuda no momento que esse código foi criado
os.environ["BNB_CUDA_VERSION"] = "130"

st.set_page_config(
    page_title="Nemotron Multi-Mode ChatBot",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Reduzir o espaçamento do topo da barra lateral */
    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
    }
    [data-testid="stSidebarHeader"] {
        display: none !important;
    }
    .metric-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px 15px;
        margin-top: 5px;
        margin-bottom: 10px;
    }
    .sidebar-label {
        font-size: 0.95rem;
        font-weight: 600;
        padding-top: 6px;
    }
    .stDeployButton {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_model_config" not in st.session_state:
    st.session_state.current_model_config = {
        "repo_name": "",
        "quantization": "",
        "use_lora": False
    }

if "last_loaded_config" not in st.session_state:
    st.session_state.last_loaded_config = {
        "repo_name": "",
        "quantization": "",
        "use_lora": False
    }

if "model_unloaded" not in st.session_state:
    st.session_state.model_unloaded = False

# Essa limpeza é essencial para evitar erros de OOM (Out of Memory) na GPU ao alterar as configurações de execução do modelo
def cleanup_memory():
    model = st.session_state.get("model")
    tokenizer = st.session_state.get("tokenizer")
    
    if "model" in st.session_state:
        st.session_state.model = None
        del st.session_state.model
    if "tokenizer" in st.session_state:
        st.session_state.tokenizer = None
        del st.session_state.tokenizer
        
    for key in list(globals().keys()):
        if key in ["model", "tokenizer", "base_model"]:
            globals()[key] = None

    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
        del model
        
    if tokenizer is not None:
        del tokenizer
        
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    st.session_state.current_model_config = {
        "repo_name": "",
        "quantization": "",
        "use_lora": False
    }

def load_model(repo_name, quant_mode, use_lora):
    config = st.session_state.current_model_config
    
    if (config["repo_name"] == repo_name and 
        config["quantization"] == quant_mode and 
        config["use_lora"] == use_lora and 
        "model" in st.session_state and 
        "tokenizer" in st.session_state):
        return st.session_state.tokenizer, st.session_state.model
        
    if "model" in st.session_state or "tokenizer" in st.session_state:
        cleanup_memory()
        
    if repo_name == "nvidia/Nemotron-Labs-Diffusion-3B" and (Path.cwd() / "hf_repo").exists():
        model_path = str(Path.cwd() / "hf_repo")
        is_local = True
    else:
        model_path = repo_name
        is_local = False
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if quant_mode == "4-bit (NF4)":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            base_model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto"
            )
        elif quant_mode == "8-bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )
            base_model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            base_model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
        if use_lora:
            lora_path = Path(model_path) / "linear_spec_lora"
            if lora_path.exists():
                model = PeftModel.from_pretrained(base_model, str(lora_path)).eval().model
            elif not is_local and repo_name == "nvidia/Nemotron-Labs-Diffusion-3B":
                model = PeftModel.from_pretrained(base_model, repo_name, subfolder="linear_spec_lora").eval().model
            else:
                st.sidebar.warning("Adaptador LoRA não encontrado. Carregando modelo base.")
                model = base_model
        else:
            model = base_model
            
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.current_model_config = {
            "repo_name": repo_name,
            "quantization": quant_mode,
            "use_lora": use_lora
        }
        st.session_state.last_loaded_config = {
            "repo_name": repo_name,
            "quantization": quant_mode,
            "use_lora": use_lora
        }
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {str(e)}")
        cleanup_memory()
        return None, None

# ==========================================
# DESIGN DA INTERFACE E BARRA LATERAL
# ==========================================
st.sidebar.title("Configurações do Nemotron")

if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    st.sidebar.markdown(f"""
    <h3 style="margin-bottom: 5px; margin-top: 0px; font-size: 1.2rem;">Status da VRAM</h3>
    <div class="metric-card">
        <div><strong>Alocada:</strong> {allocated:.2f} GB</div>
        <div><strong>Reservada:</strong> {reserved:.2f} GB</div>
        <div><strong>Pico Alocado:</strong> {peak:.2f} GB</div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown('<h3 style="margin-bottom: 5px; margin-top: 0px; font-size: 1.2rem;">Status da VRAM</h3>', unsafe_allow_html=True)
    st.sidebar.warning("CUDA não disponível no PyTorch!")

# --- SELEÇÃO DO MODELO ---
st.sidebar.markdown("### Modelo e Precisão")
model_options = [
    "nvidia/Nemotron-Labs-Diffusion-3B",
    "nvidia/Nemotron-Labs-Diffusion-8B",
    "nvidia/Nemotron-Labs-Diffusion-14B",
    "Personalizado..."
]

col_mod1, col_mod2 = st.sidebar.columns([3, 7])
with col_mod1:
    st.markdown('<div class="sidebar-label">Modelo:</div>', unsafe_allow_html=True)
with col_mod2:
    selected_option = st.selectbox("Modelo:", model_options, index=0, label_visibility="collapsed")

if selected_option == "Personalizado...":
    repo_input = st.sidebar.text_input("Nome do Repositório do HF:", value="nvidia/Nemotron-Labs-Diffusion-3B")
else:
    repo_input = selected_option

col_q1, col_q2 = st.sidebar.columns([3, 7])
with col_q1:
    st.markdown('<div class="sidebar-label">Quantização:</div>', unsafe_allow_html=True)
with col_q2:
    quant_input = st.selectbox(
        "Quantização:",
        ["4-bit (NF4) [Recomendado]", "8-bit", "Precisão Total (BF16)"],
        index=0,
        label_visibility="collapsed"
    )
quant_mode = "4-bit (NF4)" if "4-bit" in quant_input else ("8-bit" if "8-bit" in quant_input else "BF16")

use_lora_input = st.sidebar.checkbox(
    "Ativar Adaptador LoRA Especulativo (se disponível)",
    value=True if repo_input == "nvidia/Nemotron-Labs-Diffusion-3B" else False
)

last_cfg = st.session_state.last_loaded_config
if (last_cfg["repo_name"] != repo_input or 
    last_cfg["quantization"] != quant_mode or 
    last_cfg["use_lora"] != use_lora_input):
    st.session_state.model_unloaded = False

# MODO DE GERAÇÃO
st.sidebar.markdown("### Configurações de Inferência")
gen_mode = st.sidebar.selectbox(
    "Modo de Geração:",
    [
        "Linear Self-Speculation (Especulativo)",
        "AR (Autoregressivo Padrão)",
        "dLM (Diffusion Paralelo)"
    ],
    index=0
)

max_tokens = st.sidebar.slider("Máximo de novos tokens:", min_value=32, max_value=1024, value=512, step=32)
block_len = st.sidebar.slider("Tamanho do Bloco (block_length):", min_value=2, max_value=64, value=32, step=2)
threshold = st.sidebar.slider("Limiar (threshold):", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
temp = st.sidebar.slider("Temperatura:", min_value=0.0, max_value=1.5, value=0.0, step=0.1)

if "dLM" in gen_mode:
    if max_tokens % block_len != 0:
        adjusted_tokens = (max_tokens // block_len) * block_len
        if adjusted_tokens == 0:
            adjusted_tokens = block_len
        st.sidebar.info(f"Ajustado 'Máximo de tokens' para {adjusted_tokens} para ser múltiplo de {block_len} (exigido no modo dLM).")
        max_tokens = adjusted_tokens

st.sidebar.markdown("### Ações")
col_clean1, col_clean2 = st.sidebar.columns(2)
with col_clean1:
    if st.button("Limpar Histórico", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
with col_clean2:
    if st.session_state.model_unloaded:
        if st.button("Carregar Modelo", type="primary", use_container_width=True):
            st.session_state.model_unloaded = False
            st.rerun()
    else:
        if st.button("Liberar VRAM", use_container_width=True):
            cleanup_memory()
            st.session_state.model_unloaded = True
            st.success("VRAM limpa!")
            time.sleep(1)
            st.rerun()

# ==========================================
# CARREGAMENTO DO MODELO (SEÇÃO ATIVA)
# ==========================================
if not st.session_state.model_unloaded:
    with st.spinner("Carregando o modelo e preparando pesos na GPU..."):
        tokenizer, model = load_model(repo_input, quant_mode, use_lora_input)
else:
    tokenizer, model = None, None

# ==========================================
# ÁREA PRINCIPAL DO CHAT
# ==========================================
st.title("Nemotron Multi-Mode Chat")
st.markdown(f"**Modelo Ativo:** `{repo_input}` | **Precisão:** `{quant_mode}` | **Modo de Geração:** `{gen_mode}`")

if model is None or tokenizer is None:
    st.warning("O modelo não está carregado. Clique em 'Carregar Modelo' na barra lateral para iniciar.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metrics" in message:
            st.caption(message["metrics"])

if user_input := st.chat_input("Digite sua mensagem..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    
    try:
        prompt_formatted = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_formatted, return_tensors='pt').input_ids.to('cuda')
    except Exception as e:
        st.error(f"Erro ao formatar histórico de mensagens: {str(e)}")
        st.stop()
        
    with st.chat_message("assistant"):
        with st.spinner("Nemotron respondendo..."):
            start_time = time.time()
            
            try:
                if "AR" in gen_mode:
                    out_ids, nfe = model.ar_generate(
                        prompt_ids,
                        max_new_tokens=max_tokens,
                        temperature=temp,
                        eos_token_id=tokenizer.eos_token_id
                    )
                elif "dLM" in gen_mode:
                    out_ids, nfe = model.generate(
                        prompt_ids,
                        max_new_tokens=max_tokens,
                        block_length=block_len,
                        threshold=threshold,
                        temperature=temp,
                        eos_token_id=tokenizer.eos_token_id
                    )
                else:
                    out_ids, nfe = model.linear_spec_generate(
                        prompt_ids,
                        max_new_tokens=max_tokens,
                        block_length=block_len,
                        threshold=threshold,
                        temperature=temp,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                end_time = time.time()
                
                new_tokens_ids = out_ids[0, prompt_ids.shape[1]:]
                response_text = tokenizer.decode(new_tokens_ids, skip_special_tokens=True)
                
                duration = end_time - start_time
                num_tokens = len(new_tokens_ids)
                tks = num_tokens / duration if duration > 0 else 0
                metrics_text = f"{num_tokens} tokens | {duration:.2f}s | {tks:.2f} TK/s | NFE={nfe}"
                
                st.markdown(response_text)
                st.caption(metrics_text)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "metrics": metrics_text
                })
                
            except Exception as e:
                st.error(f"Erro na inferência: {str(e)}")
                if "out of memory" in str(e).lower():
                    st.warning("Estouro de memória da GPU detectado! Liberando VRAM automaticamente...")
                    cleanup_memory()
                    st.session_state.model_unloaded = True
                    st.rerun()
                else:
                    st.info("Caso ocorra estouro de memória (OOM), clique no botão 'Liberar VRAM' na barra lateral.")
