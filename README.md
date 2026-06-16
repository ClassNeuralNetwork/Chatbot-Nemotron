# Chatbot Utilizando modelo Nemotron-Labs-Diffusion-3B

Este projeto implementa um chatbot em português utilizando o modelo `Nemotron-Labs-Diffusion-3B` da NVIDIA. O diferencial desse projeto é a exploração prática de três modos de inferência distintos (AR, DLM e SPEC), integrados em uma interface amigável com suporte nativo a otimização de memória.

## Estrutura de Pastas

A organização do repositório está dividida entre o ambiente de experimentação (notebooks) e os scripts de execução da aplicação web:

```text
.
├── notebooks/
│   ├── Chat_com_o_modelo.ipynb        # Ambiente interativo para testes no Colab
│   └── Testes_com_o_modelo.ipynb      # Avaliação de métricas e requisições puras do modelo
├── scripts/
│   ├── app_streamlit.py               # Código-fonte da interface do chatbot
│   ├── benchmarks.py                  # Script para rodar os comparativos de desempenho
│   ├── dados_benchmark.json           # Saída dos dados brutos de benchmark
│   ├── plotar_graficos.py             # Script para geração das visualizações
│   ├── speed_by_category.png          # Gráfico: velocidade de geração (tok/s)
│   └── speedup_by_category.png        # Gráfico: taxa de aceleração
├── .gitignore
├── LICENSE
├── README.md
└── Relatórios_Explicativo.pdf         # Documentação completa com as análises do modelo
```

## Ferramentas Utilizadas

O ecossistema do projeto foi construído para lidar com as altas exigências de hardware do LLM:

*   **Modelo de IA:** `nvidia/Nemotron-Labs-Diffusion-3B` (3 bilhões de parâmetros).
*   **Interface:** `streamlit` para gerenciamento dinâmico da conversa e do painel de controle lateral.
*   **Processamento e Otimização:**
    *   `transformers` e `accelerate`: Responsáveis pelo carregamento do modelo, tokenizador e distribuição automática dos pesos na GPU.
    *   `torch`: Para manipulação de tensores, cálculo de inferência e monitoramento/limpeza de cache (`torch.cuda.empty_cache()`).
    *   `bitsandbytes`: Aplicado para quantizar o modelo em 4-bit (NF4) ou 8-bit, viabilizando a execução sem estourar a memória.
    *   `peft`: Utilizado para integrar o adaptador LoRA Especulativo (`linear_spec_lora`) ao modelo base.

## Como Executar

A interface simplifica o uso, mas requer que as dependências estejam devidamente instaladas.

1. **Instale as bibliotecas base:**
   ```bash
   pip install transformers accelerate torch bitsandbytes peft streamlit
   ```

2. **Inicie a aplicação:**
   A partir da raiz do repositório, execute o script principal:
   ```bash
   streamlit run scripts/app_streamlit.py
   ```

3. **Utilização do Painel:**
   * Na barra lateral esquerda, configure o modelo e o nível de quantização (ex: 4-bit NF4).
   * Selecione o modo de geração desejado (AR, dLM ou SPEC).
   * Em caso de estouro de memória (OOM), clique em **"Liberar VRAM"**. O sistema executará uma rotina segura que limpa as variáveis e o cache CUDA sem derrubar a aplicação.

## Modos de Geração e Resultados

O sistema permite alternar entre três arquiteturas de processamento de tokens. O desempenho foi avaliado através de testes padronizados:

*   **AR (Autoregressivo):** Processa tokens sequencialmente, da esquerda para a direita. Ideal para alta concorrência em nuvem.
*   **dLM (Diffusion Paralelo):** Processa tokens em blocos. O sistema ajusta automaticamente o limite de novos tokens para que seja múltiplo do tamanho do bloco (ex: 32).
*   **SPEC (Especulativo):** Combina os modos anteriores, gerando tokens via difusão e validando a coerência contextualmente via AR.

### Resumo Comparativo Médio (Google Colab)

A tabela abaixo apresenta os resultados médios dos testes para cada modo:

| Modo | NFE médio | Tempo médio | Tokens médio | tok/s médio |
| :--- | :--- | :--- | :--- | :--- |
| **AR** | 256.0 | 11.29s | 256.0 | 22.7 |
| **DLM** | 218.0 | 51.41s | 256.0 | 5.0 |
| **SPEC** | 171.4 | 41.09s | 256.8 | 6.3 |

**Conclusão da Análise:** 
Todos os modos produziram respostas adequadas e com semântica equivalente. O modo **AR** apresentou a melhor velocidade de resposta (22.7 tok/s), o que indica que as limitações de hardware do Google Colab não beneficiaram o processamento paralelo exigido pelo modo DLM neste ambiente de teste.