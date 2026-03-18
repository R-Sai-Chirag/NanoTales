import json
import time
from pathlib import Path
from model import GPT
from model_v2 import GPT2
from config import GPTConfig
import torch

CHECKPOINT="checkpoints_muon/best.pt"
VOCAB_PATH="data/tinystories/vocab.json"
PROMPT="One day Lilly found a glowing door in the forest"
MAX_TOKENS=492
TOP_K=40
NUM_STORIES=6
TEMPERATURE=0.8
USE_KV_CACHE=True

def load_vocab(path):
    with open(path,mode="r",encoding="utf-8") as f:
        token_to_id=json.load(f)
    id_to_token={v:k for k,v in token_to_id.items()}
    return token_to_id,id_to_token

import tiktoken

def encode(text, token_to_id):
    enc = tiktoken.get_encoding("gpt2")
    gpt2_ids = enc.encode(text)
    
    ids = []
    for gpt2_id in gpt2_ids:
        key = str(gpt2_id)  
        if key in token_to_id:
            ids.append(token_to_id[key])
        else:
            print(f"[Warning] GPT-2 token {gpt2_id} not in vocab, skipping")
    return ids


def decode(ids, id_to_token):
    """Decode: NanoTales IDs → GPT-2 IDs → text."""
    enc = tiktoken.get_encoding("gpt2")

    gpt2_ids = [int(id_to_token[i]) for i in ids if i in id_to_token]
    return enc.decode(gpt2_ids)


def load_model(checkpoint_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)


    model_config = GPTConfig(**checkpoint["model_config"])

    CHECKPOINT = torch.load(checkpoint_path, map_location=device)

    version = CHECKPOINT.get("model_version", "1.0")

    if version == "2.0":
        print("[NanoTales] Detected v2.0 — SwiGLU + Muon config")
        model = GPT2(model_config).to(device)
    else:
        print("[NanoTales] Detected v1.0 — standard config")
        model = GPT(model_config).to(device)


    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"[NanoTales] Loaded checkpoint from step {checkpoint['step']}")
    print(f"[NanoTales] Val loss at checkpoint: {checkpoint['loss']:.4f}")
    print(f"[NanoTales] Running on: {device}")
    

    return model, device

def generate_story(model, device, token_to_id, id_to_token, prompt, max_tokens, temperature, top_k,use_kv_cache):
    # encode prompt
    prompt_ids = encode(prompt, token_to_id)
    if len(prompt_ids) == 0:
        print("[Error] No valid tokens in prompt")
        return ""

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)


    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k,use_kv_cache=use_kv_cache)

    generated_ids = output[0].tolist()
    story = decode(generated_ids, id_to_token)
    return story

def main():
    print(f"[NanoTales] Loading vocab from {VOCAB_PATH}...")
    token_to_id, id_to_token = load_vocab(VOCAB_PATH)
    print(f"[NanoTales] Vocab size: {len(token_to_id):,} tokens")

    model, device = load_model(CHECKPOINT)
    
    if model.config.use_rope and model.config.use_gqa:
        print(f"[NanoTales_GQA_ROPE] ...")
    elif model.config.use_rope:
        print(f"[NanoTales_ROPE] ...")
    else:
        print(f"[NanoTales_BASE] ...")

    print(f"\n[NanoTales] Prompt: '{PROMPT}'")
    print(f"[NanoTales] Generating {NUM_STORIES} stories...\n")
    print("=" * 60)

    for i in range(NUM_STORIES):
        t1=time.time()
        story = generate_story(
            model, device,
            token_to_id, id_to_token,
            PROMPT, MAX_TOKENS, TEMPERATURE, TOP_K,
            use_kv_cache=USE_KV_CACHE
        )
        t2=time.time()
        elapsed=t2-t1
        tokens_per_sec=MAX_TOKENS/elapsed
        print(f"\nStory {i+1}:\n")
        print(story)
        print(f"\n{MAX_TOKENS} tokens generated in {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()