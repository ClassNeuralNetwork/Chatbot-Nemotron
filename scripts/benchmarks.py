import os
import sys
import json
import torch
import gc
import time
import types
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# necessario se usando windows e cuda 13.2
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
os.environ["BNB_CUDA_VERSION"] = "130"


# =================================================================
# VARIÁVEIS DE CONFIGURAÇÃO DO BENCHMARK
# =================================================================
N_ITERATIONS = 3            # Número de execuções para cálculo da média aritmética (n)
MAX_NEW_TOKENS = 150        # Limite máximo de novos tokens a serem gerados
BLOCK_LENGTH_1 = 32         # Primeiro tamanho de bloco de especulação a ser testado
BLOCK_LENGTH_2 = 16         # Segundo tamanho de bloco de especulação a ser testado
RUN_4BIT = True             # Executar benchmark do modelo em 4-bits (NF4)
RUN_8BIT = True             # Executar benchmark do modelo em 8-bits (Int8)
JSON_OUTPUT_FILE = "dados_benchmark.json"

local_repo_dir = Path.cwd() / "nvidia/Nemotron-Labs-Diffusion-3B"
lora_subfolder = "linear_spec_lora"

# Prompts para os testes
TEST_PROMPTS = [
    {
        "category": "Chat/Conversation",
        "prompt": "Olá, tudo bem? Gostaria de saber como você está e se pode me ajudar a planejar um roteiro de viagem de 3 dias para o Rio de Janeiro."
    },
    {
        "category": "Coding/Programming",
        "prompt": "Escreva uma função em Python para verificar se uma palavra é um palíndromo (lê-se igual de trás para frente) e adicione testes simples."
    },
    {
        "category": "Roleplay/Creative",
        "prompt": "Você é um mago antigo em uma taverna medieval. Um jovem aventureiro se aproxima pedindo conselhos sobre uma floresta misteriosa. O que você diz?"
    },
    {
        "category": "Logic/Reasoning",
        "prompt": "Se Maria tem 5 maçãs e come 2, depois ganha o dobro do que sobrou de seu pai, com quantas maçãs Maria termina?"
    }
]

def patch_linear_spec_generate(model):
    def linear_spec_generate_tracked(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        mask_token_id: int | None = None,
        eos_token_id: int | None = None,
        max_thinking_tokens: int | None = None,
        end_think_token_id: int | None = None,
        threshold: float = 0.0,
    ):
        if prompt_ids.shape[0] != 1:
            raise ValueError("Linear speculative decoding requires batch_size == 1")

        token_mask_id = mask_token_id if mask_token_id is not None else self.config.mask_token_id
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)

        device = prompt_ids.device

        def _set_diffusion_lm(val: bool):
            for layer in self.encoder.layers:
                if hasattr(layer.self_attn, "diffusion_lm"):
                    layer.self_attn.diffusion_lm = val

        def _toggle_adapters(enable: bool):
            for module in self.modules():
                if hasattr(module, "_disable_adapters"):
                    module._disable_adapters = not enable

        module_name = self.__class__.__module__
        model_module = sys.modules[module_name]
        _crop_dynamic_cache = getattr(model_module, "_crop_dynamic_cache")
        from transformers.cache_utils import DynamicCache
        
        # Prefill (causal, LoRA OFF).
        _set_diffusion_lm(False)
        _toggle_adapters(False)
        enc_out = self.encoder(
            input_ids=prompt_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
            use_causal_mask=True,
        )
        past_key_values = enc_out.past_key_values
        last_logit = self.diffusion_head(enc_out.last_hidden_state[:, -1:, :]).squeeze(1)
        nfe = 1

        if temperature > 0:
            next_token = torch.multinomial(torch.softmax(last_logit / temperature, dim=-1), num_samples=1)
        else:
            next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            return torch.cat([prompt_ids, next_token], dim=1), nfe, 1.0

        generated = [next_token]
        total_gen = 1

        acceptance_rates = []

        while total_gen < max_new_tokens:
            cache_len = past_key_values.get_seq_length()

            block = torch.full((1, block_length), token_mask_id, dtype=torch.long, device=device)
            block[0, 0] = next_token.item()

            # Draft phase (bidirectional, LoRA ON)
            _set_diffusion_lm(True)
            _toggle_adapters(True)
            while True:
                is_mask = block == token_mask_id
                if not is_mask.any():
                    break

                enc_out = self.encoder(input_ids=block, past_key_values=past_key_values, use_cache=False)
                nfe += 1

                draft_logits = self.diffusion_head(enc_out.last_hidden_state)

                if temperature > 0:
                    draft_probs = torch.softmax(draft_logits / temperature, dim=-1)
                    draft_tokens = torch.multinomial(
                        draft_probs.view(-1, draft_probs.shape[-1]), num_samples=1
                    ).view(1, block_length)
                else:
                    draft_tokens = draft_logits.argmax(dim=-1)
                    draft_probs = torch.softmax(draft_logits, dim=-1)

                if threshold > 0:
                    draft_conf = torch.gather(draft_probs, -1, draft_tokens.unsqueeze(-1)).squeeze(-1)
                    draft_conf = torch.where(is_mask, draft_conf, -torch.inf)
                    unmask = draft_conf >= threshold
                    if not unmask.any():
                        best_idx = draft_conf.view(-1).argmax()
                        unmask = torch.zeros_like(is_mask, dtype=torch.bool)
                        unmask.view(-1)[best_idx] = True
                    block[unmask] = draft_tokens[unmask]
                else:
                    block[is_mask] = draft_tokens[is_mask]
                    break

            # Verify phase (causal, LoRA OFF).
            _set_diffusion_lm(False)
            _toggle_adapters(False)
            enc_out = self.encoder(
                input_ids=block,
                past_key_values=past_key_values,
                use_cache=True,
                use_causal_mask=True,
            )
            past_key_values = enc_out.past_key_values
            nfe += 1

            verify_logits = self.diffusion_head(enc_out.last_hidden_state)
            if temperature > 0:
                ar_tokens = torch.multinomial(
                    torch.softmax(verify_logits / temperature, dim=-1).view(-1, verify_logits.shape[-1]),
                    num_samples=1,
                ).view(1, block_length)
            else:
                ar_tokens = verify_logits.argmax(dim=-1)

            # Accept consecutive matches
            accepted = 0
            for i in range(block_length - 1):
                if ar_tokens[0, i].item() == block[0, i + 1].item():
                    accepted += 1
                else:
                    break
            accepted += 1

            # Store acceptance rate of this step (drafted vs accepted)
            drafted_count = block_length - 1
            accepted_drafted = accepted - 1
            step_acc_rate = accepted_drafted / drafted_count if drafted_count > 0 else 1.0
            acceptance_rates.append(step_acc_rate)

            accepted_toks = ar_tokens[:, :accepted]
            generated.append(accepted_toks)
            total_gen += accepted

            _crop_dynamic_cache(past_key_values, cache_len + accepted)
            next_token = ar_tokens[:, accepted - 1 : accepted]

            if eos_token_id is not None:
                eos_pos = (accepted_toks[0] == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    first_eos = eos_pos[0].item()
                    generated[-1] = accepted_toks[:, : first_eos + 1]
                    total_gen = total_gen - accepted + first_eos + 1
                    break

            if end_think_token_id is not None and max_thinking_tokens is not None:
                if total_gen > max_thinking_tokens:
                    all_gen = torch.cat(generated, dim=1)
                    if not (all_gen == end_think_token_id).any():
                        next_token = torch.tensor([[end_think_token_id]], device=device)

            if total_gen >= max_new_tokens:
                break

        all_generated = torch.cat(generated, dim=1)
        output_ids = torch.cat([prompt_ids, all_generated], dim=1)
        
        mean_acc_rate = sum(acceptance_rates) / len(acceptance_rates) if acceptance_rates else 0.0
        return output_ids, nfe, mean_acc_rate

    # Monkeypatch
    model.linear_spec_generate = types.MethodType(linear_spec_generate_tracked, model)


def run_single_setup(tokenizer, model, active_model, mode_type, prompt_ids, max_tokens, n_iterations, block_length):
    total_tks = 0
    total_duration = 0
    total_tokens = 0
    total_nfe = 0
    total_acc_rate = 0.0
    
    for _ in range(n_iterations):
        gc.collect()
        torch.cuda.empty_cache()
        
        start_time = time.time()
        acc_rate = 1.0
        
        if mode_type == "ar":
            out_ids, nfe = active_model.ar_generate(prompt_ids, max_new_tokens=max_tokens)
        else:
            out_ids, nfe, acc_rate = active_model.linear_spec_generate(
                prompt_ids,
                max_new_tokens=max_tokens,
                block_length=block_length,
                eos_token_id=tokenizer.eos_token_id
            )
            
        end_time = time.time()
        
        duration = end_time - start_time
        new_tokens_ids = out_ids[0, prompt_ids.shape[1]:]
        num_tokens = len(new_tokens_ids)
        tks = num_tokens / duration if duration > 0 else 0
        
        total_tks += tks
        total_duration += duration
        total_tokens += num_tokens
        total_nfe += nfe
        total_acc_rate += acc_rate
        
    return {
        "tks": total_tks / n_iterations,
        "duration": total_duration / n_iterations,
        "tokens": total_tokens / n_iterations,
        "nfe": total_nfe / n_iterations,
        "acc_rate": total_acc_rate / n_iterations
    }


def benchmark_model_config(quant_mode, n_iterations):
    print(f"\n=============================================================")
    print(f" CARREGANDO MODELO EM MODO: {quant_mode} ")
    print(f"=============================================================")
    
    if quant_mode == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        theoretical_size = 1.9  # GB
    else:  # 8bit
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True
        )
        theoretical_size = 3.8  # GB
        
    tokenizer = AutoTokenizer.from_pretrained(str(local_repo_dir), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(local_repo_dir),
        trust_remote_code=True,
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    vram_allocated = 0.0
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM Alocada para pesos do modelo: {vram_allocated:.2f} GB")
        
    patch_linear_spec_generate(model)
    
    print("Carregando adaptador LoRA...")
    lora_model = PeftModel.from_pretrained(
        model,
        str(local_repo_dir),
        subfolder=lora_subfolder,
        is_trainable=False
    ).eval()
    spec_lora_model = lora_model.model
    
    config_results = []
    
    for item in TEST_PROMPTS:
        category = item["category"]
        prompt_text = item["prompt"]
        
        print(f"\nPrompt: [{category}] '{prompt_text[:50]}...'")
        
        history = [{"role": "user", "content": prompt_text}]
        chat_prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(chat_prompt, return_tensors="pt").input_ids.to("cuda")
        
        modes_to_test = {
            "AR (Padrão)": (model, "ar", 32),
            "Linear Spec (Sem LoRA)": (model, "spec", BLOCK_LENGTH_1),
            "Linear Spec (Com LoRA - Block 32)": (spec_lora_model, "spec", BLOCK_LENGTH_1),
            "Linear Spec (Com LoRA - Block 16)": (spec_lora_model, "spec", BLOCK_LENGTH_2)
        }
        
        prompt_res = {}
        for mode_name, (active_model, mode_type, block_len) in modes_to_test.items():
            res = run_single_setup(
                tokenizer, model, active_model, mode_type, prompt_ids, 
                MAX_NEW_TOKENS, n_iterations, block_len
            )
            print(f"  -> {mode_name:32} | {res['tks']:5.2f} TK/s | Tokens: {res['tokens']:5.1f} | NFE: {res['nfe']:5.1f} | Acc Rate: {res['acc_rate']*100:5.1f}%")
            prompt_res[mode_name] = res
            
        config_results.append({
            "category": category,
            "prompt": prompt_text,
            "modes": prompt_res
        })
        
    del lora_model, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "results": config_results,
        "vram": vram_allocated,
        "theoretical_size": theoretical_size
    }


def main():
    print("=================================================================")
    print(" INICIANDO EXECUÇÃO DO BENCHMARK ")
    print("=================================================================")
    print(f"Configurações: n={N_ITERATIONS} | max_tokens={MAX_NEW_TOKENS} | blocks=[{BLOCK_LENGTH_1}, {BLOCK_LENGTH_2}]")
    
    benchmark_data = {
        "config": {
            "n_iterations": N_ITERATIONS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "block_length_1": BLOCK_LENGTH_1,
            "block_length_2": BLOCK_LENGTH_2
        },
        "results": {}
    }
    
    if RUN_8BIT:
        res_8bit = benchmark_model_config("8bit", N_ITERATIONS)
        benchmark_data["results"]["8bit"] = res_8bit
        
    if RUN_4BIT:
        # Garantir limpeza absoluta de VRAM
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)
        
        res_4bit = benchmark_model_config("4bit", N_ITERATIONS)
        benchmark_data["results"]["4bit"] = res_4bit
        
    # Exportar os dados brutos para JSON
    output_path = Path.cwd() / JSON_OUTPUT_FILE
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=4, ensure_ascii=False)
        
    print("\n" + "="*65)
    print(f"BENCHMARK CONCLUÍDO! DADOS SALVOS EM: {JSON_OUTPUT_FILE}")
    print("="*65)


if __name__ == "__main__":
    main()
