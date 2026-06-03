import json
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def load_data(json_file="dados_benchmark.json"):
    path = Path.cwd() / json_file
    if not path.exists():
        print(f"Erro: O arquivo '{json_file}' não foi encontrado. Execute 'benchmarks.py' primeiro.")
        sys.exit(1)
        
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_charts():
    data = load_data()
    results = data["results"]
    
    # Mapeamento de categorias curtas para os rótulos do gráfico
    category_labels = {
        "Chat/Conversation": "Conversação",
        "Coding/Programming": "Programação",
        "Roleplay/Creative": "Criativo/Roleplay",
        "Logic/Reasoning": "Lógica/Raciocínio"
    }
    
    categories = list(category_labels.keys())
    display_categories = [category_labels[c] for c in categories]
    
    modes_keys = [
        "AR (Padrão)",
        "Linear Spec (Sem LoRA)",
        "Linear Spec (Com LoRA - Block 32)",
        "Linear Spec (Com LoRA - Block 16)"
    ]
    
    modes_labels = [
        "AR (Padrão)",
        "Spec (Sem LoRA)",
        "Spec LoRA B32",
        "Spec LoRA B16"
    ]
    
    # Dicionários para guardar TK/s por categoria e modo
    tks_4bit_data = {m: [] for m in modes_keys}
    tks_8bit_data = {m: [] for m in modes_keys}
    
    # Extrair dados de 4-bit
    if "4bit" in results:
        prompts_res = results["4bit"]["results"]
        # Organiza dados na ordem das categorias definidas
        for cat in categories:
            prompt_data = next(p for p in prompts_res if p["category"] == cat)
            for m in modes_keys:
                tks_4bit_data[m].append(prompt_data["modes"][m]["tks"])
                
    # Extrair dados de 8-bit
    if "8bit" in results:
        prompts_res = results["8bit"]["results"]
        for cat in categories:
            prompt_data = next(p for p in prompts_res if p["category"] == cat)
            for m in modes_keys:
                tks_8bit_data[m].append(prompt_data["modes"][m]["tks"])

    # Configuração de Estilo do Matplotlib
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({'font.size': 11, 'figure.titlesize': 14})
    
    # =================================================================
    # GRÁFICO 1: Velocidade por Categoria / Tipo de Conversa (4-bit vs 8-bit)
    # =================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10), sharex=False)
    
    x = np.arange(len(display_categories))
    width = 0.2
    
    # Cores modernas para as barras
    colors = ["#9ca3af", "#60a5fa", "#2563eb", "#059669"] # Cinza, Azul claro, Azul escuro, Verde escuro
    
    # Plot 4-bit
    for idx, m_key in enumerate(modes_keys):
        offset = (idx - 1.5) * width
        rects = ax1.bar(x + offset, tks_4bit_data[m_key], width, label=modes_labels[idx], color=colors[idx])
        ax1.bar_label(rects, padding=3, fmt='%.1f', fontsize=9)
        
    ax1.set_ylabel('Tokens por Segundo (TK/s)')
    ax1.set_title('Inference Speed by Conversation Type - 4-bit NF4 Quantization (VRAM ~3.2GB)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(display_categories)
    ax1.set_ylim(0, 32)
    ax1.legend()
    
    # Plot 8-bit
    for idx, m_key in enumerate(modes_keys):
        offset = (idx - 1.5) * width
        rects = ax2.bar(x + offset, tks_8bit_data[m_key], width, label=modes_labels[idx], color=colors[idx])
        ax2.bar_label(rects, padding=3, fmt='%.1f', fontsize=9)
        
    ax2.set_ylabel('Tokens por Segundo (TK/s)')
    ax2.set_title('Inference Speed by Conversation Type - 8-bit Int8 Quantization (VRAM ~4.7GB)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(display_categories)
    ax2.set_ylim(0, 16)
    ax2.legend()
    
    fig.suptitle('Nemotron-Labs-Diffusion-3B: Desempenho por Tipo de Conversa na GPU', y=0.98, fontweight='bold')
    fig.tight_layout()
    
    chart_path = Path.cwd() / "speed_by_category.png"
    plt.savefig(chart_path, dpi=150)
    plt.close(fig)
    
    # =================================================================
    # GRÁFICO 2: Fator de Aceleração (Speedup) por Categoria
    # =================================================================
    fig2, ax_sp = plt.subplots(figsize=(10, 5.5))
    
    # Calcular speedup do Spec LoRA Block 32 em relação ao AR Padrão
    speedup_4bit = []
    speedup_8bit = []
    
    for i in range(len(categories)):
        ar_4 = tks_4bit_data["AR (Padrão)"][i]
        spec_4 = tks_4bit_data["Linear Spec (Com LoRA - Block 32)"][i]
        speedup_4bit.append(spec_4 / ar_4 if ar_4 > 0 else 0)
        
        ar_8 = tks_8bit_data["AR (Padrão)"][i]
        spec_8 = tks_8bit_data["Linear Spec (Com LoRA - Block 32)"][i]
        speedup_8bit.append(spec_8 / ar_8 if ar_8 > 0 else 0)
        
    width_sp = 0.35
    rects1 = ax_sp.bar(x - width_sp/2, speedup_4bit, width_sp, label='Aceleração 4-bit (Spec B32 / AR)', color='#3b82f6')
    rects2 = ax_sp.bar(x + width_sp/2, speedup_8bit, width_sp, label='Aceleração 8-bit (Spec B32 / AR)', color='#8b5cf6')
    
    # Adicionar linha horizontal em 1.0x para referência
    ax_sp.axhline(1.0, color='red', linestyle='--', linewidth=1, label='Linha Base AR (1.0x)')
    
    ax_sp.set_ylabel('Fator de Aceleração (Speedup x)')
    ax_sp.set_title('Ganho de Desempenho da Decodificação Especulativa por Categoria de Conversa')
    ax_sp.set_xticks(x)
    ax_sp.set_xticklabels(display_categories)
    ax_sp.set_ylim(0, 3.0)
    ax_sp.legend()
    
    ax_sp.bar_label(rects1, padding=3, fmt='%.2fx')
    ax_sp.bar_label(rects2, padding=3, fmt='%.2fx')
    
    fig2.tight_layout()
    chart_sp_path = Path.cwd() / "speedup_by_category.png"
    plt.savefig(chart_sp_path, dpi=150)
    plt.close(fig2)
    
    print("Gráficos de categoria gerados com sucesso:")
    print(f" - {chart_path.name}")
    print(f" - {chart_sp_path.name}")

if __name__ == "__main__":
    generate_charts()
