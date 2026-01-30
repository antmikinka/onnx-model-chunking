from transformers import AutoTokenizer
import os

tokenizer_path = 'C:/Users/antmi/.cache/huggingface/hub/models--amd--Qwen2.5-0.5B-Instruct-quantized_int4-float16-cpu-onnx/snapshots/7d826c831528165f5289bbb0975894b078f1da7b'
prompt_file = 'MASTER_POC/stress_prompt.txt'

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
with open(prompt_file, 'r', encoding='utf-8') as f:
    base_text = f.read()

base_ids = tokenizer.encode(base_text)
current_len = len(base_ids)
target_len = 1024

if current_len < target_len:
    # Add padding tokens
    padding_needed = target_len - current_len
    # We use a neutral word to pad as [PAD] might be filtered or behave weirdly
    pad_text = " [STRESS_TEST_PADDING]" * padding_needed
    final_ids = tokenizer.encode(base_text + pad_text, truncation=True, max_length=target_len)
    final_text = tokenizer.decode(final_ids)
    
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(final_text)
    print(f"Padded prompt to {len(tokenizer.encode(final_text))} tokens.")
else:
    print(f"Prompt is already {current_len} tokens.")
