import os
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "0"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = "./.tc"
import json
import numpy as np
import math
import time
import torch
from pathlib import Path
from contextlib import nullcontext
from model_v3 import GPT3
from config import GPTConfig,TrainConfig
from tqdm.auto import tqdm
from muon import MuonWithAuxAdam

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True



os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

import torch.distributed as dist

if not dist.is_initialized():
    dist.init_process_group(
        backend="gloo",
        store=dist.FileStore("tmp_dist_store", 1),
        world_size=1,
        rank=0
    )

_MEMMAP_CACHE={}
def get_batch(split,config):
    data_path=Path(config.data_dir)/split
    chunks=sorted(data_path.glob("*.bin"))

    chunk_path=chunks[torch.randint(len(chunks),(1,)).item()]# returns 1D tensor with 1 element  → tensor([423])


    if chunk_path not in _MEMMAP_CACHE:
        _MEMMAP_CACHE[chunk_path]=np.memmap(chunk_path,dtype=np.uint16,mode="r")

    data=_MEMMAP_CACHE[chunk_path]

    idx=torch.randint(len(data)-config.block_size,(config.batch_size,))

    x=torch.stack([torch.from_numpy(data[i:i+config.block_size].astype(np.int64))for i in idx])
    y=torch.stack([torch.from_numpy(data[i+1:i+1+config.block_size].astype(np.int64))for i in idx])

    if config.device=="cuda":
        x=x.pin_memory().to(config.device,non_blocking=True)
        y=y.pin_memory().to(config.device,non_blocking=True)
    else:
        x=x.to(config.device)
        y=y.to(config.device)


    return x,y

def configure_optimizer(model, config):
    no_decay_names = {"b_res", "b_pre", "b_post", "a_pre", "a_post", "a_res"}

    hidden_parameters = [p for n, p in model.named_parameters()
                         if p.ndim >= 2
                         and "lm_head" not in n
                         and "wte" not in n
                         and not any(nd in n for nd in no_decay_names)]  # ← exclude mHC statics

    hidden_biases = [p for n, p in model.named_parameters()
                     if p.ndim < 2
                     and not any(nd in n for nd in no_decay_names)]      # ← exclude mHC statics

    non_hidden = [*model.transformer.wte.parameters()]

    mhc_static = [p for n, p in model.named_parameters()
                  if any(nd in n for nd in no_decay_names)]

    params_groups = [
        dict(params=hidden_parameters, use_muon=True, lr=config.muon_learning_rate, weight_decay=config.weight_decay),
        dict(params=hidden_biases + non_hidden + mhc_static, use_muon=False, lr=config.learning_rate,
             betas=(config.beta1, config.beta2), weight_decay=0.0),
    ]

    return MuonWithAuxAdam(param_groups=params_groups)

def get_lr(step, config):
    # Phase 1 — warmup
    if step < config.warmup_steps:
        return step / config.warmup_steps

    # Phase 2 — cosine decay
    progress = (step - config.warmup_steps) / (config.max_iters - config.warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * progress))
    return config.min_lr_ratio + (1 - config.min_lr_ratio) * coeff


def train():
    model_config=GPTConfig()
    train_config=TrainConfig()

    device=train_config.device
    device_type="cuda" if "cuda" in device else "cpu"

    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[train_config.dtype]

    if device_type=="cuda":
        ctx=torch.amp.autocast(device_type=device_type,dtype=ptdtype)
    else:
        ctx=nullcontext()
    

    model=GPT3(model_config).to(device)

    INIT_FROM = "resume" # Change this to "scratch" if you ever want to train a new model

    if INIT_FROM == "resume":
        ckpt_path = Path(train_config.out_dir) / "best.pt"
        print(f"[NanoTales] Resuming from checkpoint: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint["model"]
        
        # Clean the compiled prefixes just like we did in generate.py
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                
        model.load_state_dict(state_dict)
        
        # Override config for a gentle Fine-Tuning run
        train_config.max_iters = 3000
        train_config.warmup_steps = 100
        train_config.learning_rate = 1e-5             # Much lower than normal AdamW
        train_config.muon_learning_rate = 0.001       # Scaled down Muon LR
        
        print(f"[NanoTales] Overriding Config for SFT: {train_config.max_iters} iters, LR={train_config.learning_rate}")
        
    if train_config.compile:
        print("Compiling model... (first 3 steps will be slow)")
        model = torch.compile(
            model,
            mode="max-autotune-no-cudagraphs",
            dynamic=True   # CRITICAL for your mHC dynamic shapes
        )


    if model.config.use_rope and model.config.use_gqa and model.config.use_mhc:
        print(f"[NanoTales] Model: RoPE + RMSNorm + GQA + SwiGLU + mHC | lanes={model.config.n_lanes} | layers={model.config.n_layer} | d_model={model.config.d_model}")
        

    optimizer=configure_optimizer(model,train_config)

    Path(train_config.out_dir).mkdir(exist_ok=True,parents=True)

    best_val_loss=float("inf")
    train_losses=[]
    val_losses=[]

    print(f"[NanoTales] Starting training on {device}")
    print(f"[NanoTales] Model parameters: {model.get_num_params():,}")
    print(f"[NanoTales] Max iters: {train_config.max_iters:,}")

    for step in tqdm(range(train_config.max_iters)):
        t0=time.time()

        optimizer.zero_grad(set_to_none=True)
        accum_loss=0.0

        for micro_step in range(train_config.gradient_accumulation_steps):
            X,y=get_batch("train",train_config)

            with ctx:
                logits,loss,_=model(X,y)
                loss=loss/train_config.gradient_accumulation_steps


            loss.backward()
            accum_loss+=loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(),train_config.grad_clip)

        lr_scale=get_lr(step,train_config)

        for group in optimizer.param_groups:
            if group.get("use_muon",False):
                group["lr"]=train_config.muon_learning_rate*lr_scale
            else:
                group["lr"]=train_config.learning_rate*lr_scale

        optimizer.step()

        t1=time.time()
        dt=t1-t0

        if step % train_config.log_interval == 0:
            print(f"step {step:6d} | loss {accum_loss:.4f} | muon_lr {train_config.muon_learning_rate * lr_scale:.6f} | adamw_lr {train_config.learning_rate * lr_scale:.6f} | time {dt*1000:.0f}ms")

            if model_config.use_mhc:
                block = model.transformer.h[0]          # check first block
                with torch.no_grad():
                    # grab a sample batch just to get h_res
                    X_sample, _ = get_batch("train", train_config)
                    n = model_config.n_lanes                    # ← n_lanes not n_streams
                    x_streams = torch.zeros(X_sample.shape[0], X_sample.shape[1], n, model_config.d_model, device=device)
                    x_streams[:, :, 0, :] = model.transformer.wte(X_sample)

                    H_pre, H_post, H_res = block.mhc.get_mappings(x_streams)  # ← att_mhc, correct unpack order
                    row_sums = H_res[0, 0].sum(-1)
                    col_sums = H_res[0, 0].sum(-2)
                    print(f"  [mHC] H_res row_sums: {row_sums.tolist()}")
                    print(f"  [mHC] H_res col_sums: {col_sums.tolist()}")

        
        if step%train_config.eval_interval==0 and step!=0:
            val_loss=evaluate(model,train_config,ctx)
            val_losses.append(val_loss)
            train_losses.append(accum_loss)
            print(f"[NanoTales] step {step} | val loss {val_loss:.4f}")
            print()

            if val_loss<best_val_loss:
                best_val_loss=val_loss
                save_checkpoint(model,optimizer,step,val_loss,train_config,"best")
                print(f"[NanoTales] Best model saved — val loss {val_loss:.4f}")

        
        if step % train_config.save_interval == 0 and step != 0:
            save_checkpoint(model, optimizer, step, accum_loss, train_config, f"step_{step}")

    print(f"[NanoTales] Training complete. Best val loss: {best_val_loss:.4f}")
    return train_losses,val_losses

def save_checkpoint(model, optimizer,step, loss, config, name):
    checkpoint = {
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "step":         step,
        "loss":         loss,
        "model_config": model.config.__dict__,
        "model_version":"2.0",
    }

    path = Path(config.out_dir) / f"{name}.pt"
    torch.save(checkpoint, path)
    print(f"[NanoTales] Checkpoint saved → {path}")


@torch.no_grad()
def evaluate(model,config,ctx):
    model.eval()
    losses=torch.zeros(config.eval_iters)

    for k in range(config.eval_iters):
        X,y=get_batch("validation",config)
        with ctx:
            logits,loss,_=model(X,y)
        losses[k]=loss.item()

    val_loss=losses.mean().item()
    model.train()
    return val_loss

if __name__=="__main__":
    train_loss,val_loss=train()