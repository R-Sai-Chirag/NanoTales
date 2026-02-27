import json
from pathlib import Path
from model import GPT
from config import GPTConfig
import torch

CHECKPOINT="checkpoints/best.pt"
VOCAB_PATH="data/tinystories/vocab.json"
PROMPT="Lily did not want to share her toys"
MAX_TOKENS=200
TOP_K=40
NUM_STORIES=3
TEMPERATURE=0.9

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
    model = GPT(model_config).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"[NanoTales] Loaded checkpoint from step {checkpoint['step']}")
    print(f"[NanoTales] Val loss at checkpoint: {checkpoint['loss']:.4f}")
    print(f"[NanoTales] Running on: {device}")

    return model, device

def generate_story(model, device, token_to_id, id_to_token, prompt, max_tokens, temperature, top_k):
    # encode prompt
    prompt_ids = encode(prompt, token_to_id)
    if len(prompt_ids) == 0:
        print("[Error] No valid tokens in prompt")
        return ""

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)


    with torch.no_grad():
        output = model.generate(idx, max_new_tokens=max_tokens, temperature=temperature, top_k=top_k)

    generated_ids = output[0].tolist()
    story = decode(generated_ids, id_to_token)
    return story

def main():
    print(f"[NanoTales] Loading vocab from {VOCAB_PATH}...")
    token_to_id, id_to_token = load_vocab(VOCAB_PATH)
    print(f"[NanoTales] Vocab size: {len(token_to_id):,} tokens")

    print(f"[NanoTales] Loading model from {CHECKPOINT}...")
    model, device = load_model(CHECKPOINT)

    print(f"\n[NanoTales] Prompt: '{PROMPT}'")
    print(f"[NanoTales] Generating {NUM_STORIES} stories...\n")
    print("=" * 60)

    for i in range(NUM_STORIES):
        story = generate_story(
            model, device,
            token_to_id, id_to_token,
            PROMPT, MAX_TOKENS, TEMPERATURE, TOP_K
        )
        print(f"\nStory {i+1}:\n")
        print(story)
        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()