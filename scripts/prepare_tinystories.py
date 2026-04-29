import os
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from collections import Counter
from datasets import load_dataset
import tiktoken
import json
import argparse


def parse_args():
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train","validation"],
        help="Which data split to process."
    )

    parser.add_argument(
        "--destination",
        type=str,
        default="data/tinystories",
        help="Where to store the .bin file."
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10000,
        help="Number of most common tokens to keep."
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512000,
        help="Number of tokenks per .bin file."
    )

    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of parallel workers for tokenisation."
    )

    return parser.parse_args()


def load_data(split):
    print(f"Loading TinyStories ({split} split).")
    ds=load_dataset("roneneldan/TinyStories",split=split)
    print(f"Loaded {split} split ({len(ds):,} Stories).")
    return ds


def build_vocab(ds,vocab_size,num_proc):
    print(f"Building Vocabulary:(top {vocab_size:,} Words.)")

    encoder=tiktoken.get_encoding("gpt2")
    counter=Counter()

    def tokenize(story):
        tokens=encoder.encode_ordinary(story["text"])
        return {"tokens":tokens}
    
    tokenized_stories=ds.map(
        tokenize,
        num_proc=num_proc,
        remove_columns=["text"],
        desc="Tokenizing the stories.")
    
    for example in tqdm(tokenized_stories,desc="Counting Tokens"):
        counter.update(example["tokens"])

    # --- FIX START ---
    # Take (vocab_size - 1) to leave room for the explicit EOS token
    most_common=counter.most_common(vocab_size - 1)                                   

    vocab={}
    for new_id,(old_id,count) in enumerate(most_common):
        vocab[old_id]=new_id

    # Explicitly inject the GPT-2 EOS token (50256) at the final index
    eos_gpt2_id = encoder.eot_token
    vocab[eos_gpt2_id] = vocab_size - 1
    # --- FIX END ---

    print(f"[NanoTales] Most common token: id={most_common[0][0]}, count={most_common[0][1]:,}")
    print(f"[NanoTales] Least common token: id={most_common[-1][0]}, count={most_common[-1][1]:,}")
    print(f"[NanoTales] INJECTED EOS token: GPT-2 id={eos_gpt2_id} -> NanoTales id={vocab_size - 1}")

    return vocab,encoder,tokenized_stories


def tokenize_and_remap(story,vocab):
    remapped=[]

    for token in story["tokens"]:
        if token in vocab:
            remapped.append(vocab[token])

    # --- FIX START ---
    # Append the true EOS token using its mapped ID instead of '1'
    eos_gpt2_id = 50256 # Standard GPT-2 EOS token
    if eos_gpt2_id in vocab:
        remapped.append(vocab[eos_gpt2_id])
    # --- FIX END ---

    return {"ids":remapped , "len":len(remapped)}

def write_chunks(tokenised_ds,dir,chunk_size):

    dir=Path(dir)
    dir.mkdir(parents=True,exist_ok=True)
    buffer=[]
    buffer_len=0
    chunk_idx=0

    for example in tqdm(tokenised_ds,desc="Writing chunks"):
        buffer.extend(example["ids"])
        buffer_len+=example["len"]
    
        while buffer_len>=chunk_size:
            chunk=buffer[:chunk_size]

            chunk=np.array(chunk,dtype=np.uint16)
            out_dir=dir/f"chunk_{chunk_idx:06d}.bin"
            chunk.tofile(out_dir)
        
            buffer=buffer[chunk_size:]
            buffer_len=buffer_len-chunk_size
            chunk_idx+=1

    if buffer:
        chunk=np.array(buffer,dtype=np.uint16)
        out_dir=dir/f"chunk_{chunk_idx:06d}.bin"
        chunk.tofile(out_dir)
        chunk_idx+=1

    print(f"Written {chunk_idx} chunks to {out_dir}.")

def save_vocab(vocab, destination_path):
    vocab_path = Path(destination_path) / "vocab.json"
    
    vocab_str = {str(k): v for k, v in vocab.items()}

    Path(destination_path).mkdir(parents=True, exist_ok=True)
    
    with open(vocab_path, "w") as f:
        json.dump(vocab_str, f)
    
    print(f"[NanoTales] Vocabulary saved to {vocab_path}")


def main():
    args=parse_args()

    ds=load_data(args.split)

    vocab_path=Path(args.destination)/"vocab.json"

    if args.split=="train":
        vocab,enc,tokenised_ds=build_vocab(ds,args.vocab_size,args.num_proc)
        save_vocab(vocab,destination_path=args.destination)

    else:
        with open(vocab_path,"r") as f:
            print(f"\n[NanoTales] Loading vocabulary from {vocab_path}...")
            vocab_str = json.load(f)
            
        vocab = {int(k): v for k, v in vocab_str.items()}
        enc = tiktoken.get_encoding("gpt2")
        print(f"[NanoTales] Vocabulary loaded — {len(vocab):,} tokens")

        def tokenize(example):
            tokens = enc.encode_ordinary(example["text"])
            return {"tokens": tokens}
        
        tokenised_ds = ds.map(
            tokenize,
            num_proc=args.num_proc,
            remove_columns=["text"],
            desc="Tokenizing validation"
        )
    
    remapped_ds=tokenised_ds.map(
        lambda example: tokenize_and_remap(example,vocab),
        num_proc=args.num_proc,
        remove_columns=["tokens"],
        desc=f"Remapping {args.split}."
    )

    out_dir = Path(args.destination) / args.split
    write_chunks(remapped_ds, out_dir, args.chunk_size)
    
    print(f"\n[NanoTales] Done! {args.split} split ready for training.")

if __name__=="__main__":
    main()