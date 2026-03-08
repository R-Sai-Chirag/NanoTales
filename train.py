import os
import json
import numpy as np
import math
import time
import torch
from pathlib import Path
from contextlib import nullcontext
from torch.optim.lr_scheduler import LinearLR,CosineAnnealingLR,SequentialLR
from model import GPT
from config import GPTConfig,TrainConfig
from tqdm.auto import tqdm

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

def configure_optimizer(model,config):
    decay_params=[ p for n,p in model.named_parameters() if p.dim()>=2]
    nondecay_params=[ p for n,p in model.named_parameters() if p.dim()<2]

    params_group=[
        {"params":decay_params,"weight_decay":config.weight_decay},
        {"params":nondecay_params,"weight_decay":0.0}
    ]

    optimizer=torch.optim.AdamW(params_group,lr=config.learning_rate,betas=(config.beta1,config.beta2))

    return optimizer

def configure_scheduler(optimizer,config):
    warmup=LinearLR(optimizer,start_factor=1e-8,end_factor=1,total_iters=config.warmup_steps)

    decay=CosineAnnealingLR(optimizer,eta_min=config.min_lr,T_max=config.max_iters-config.warmup_steps)

    scheduler=SequentialLR(optimizer,schedulers=[warmup,decay],milestones=[config.warmup_steps])

    return scheduler

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
    

    model=GPT(model_config).to(device)

    if model.config.use_rope and model.config.use_gqa:
        print(f"[NanoTales] Compiling model with RoPE and GQA... {model.config.n_groups}")
        
    if train_config.compile:
        print("[NanoTales] Compiling base model...")
        model=torch.compile(model)

    optimizer=configure_optimizer(model,train_config)
    scheduler=configure_scheduler(optimizer,train_config)

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
        optimizer.step()

        scheduler.step()

        t1=time.time()
        dt=t1-t0

        if step % train_config.log_interval == 0:
            print(f"step {step:6d} | loss {accum_loss:.4f} | lr {scheduler.get_last_lr()[0]:.6f} | time {dt*1000:.0f}ms")

        
        if step%train_config.eval_interval==0 and step!=0:
            val_loss=evaluate(model,train_config,ctx)
            val_losses.append(val_loss)
            train_losses.append(accum_loss)
            print(f"[NanoTales] step {step} | val loss {val_loss:.4f}")

            if val_loss<best_val_loss:
                best_val_loss=val_loss
                save_checkpoint(model,optimizer,scheduler,step,val_loss,train_config,"best")
                print(f"[NanoTales] Best model saved — val loss {val_loss:.4f}")

        
        if step % train_config.save_interval == 0 and step != 0:
            save_checkpoint(model, optimizer, scheduler, step, accum_loss, train_config, f"step_{step}")

    print(f"[NanoTales] Training complete. Best val loss: {best_val_loss:.4f}")
    return train_losses,val_losses

def save_checkpoint(model, optimizer, scheduler, step, loss, config, name):
    checkpoint = {
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "scheduler":    scheduler.state_dict(),
        "step":         step,
        "loss":         loss,
        "model_config": model.config.__dict__,
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
            logits,loss=model(X,y)
        losses[k]=loss.item()

    val_loss=losses.mean().item()
    model.train()
    return val_loss

if __name__=="__main__":
    train_loss,val_loss=train()