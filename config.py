from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size:int=10000
    n_head:int=8
    n_layer:int=9
    d_model:int=512
    block_size:int=256
    dropout:float=0.1
    bias:bool=False
    use_rope:bool=True
    n_groups:int=4
    use_gqa:bool=True

@dataclass
class TrainConfig:
    data_dir:str="data/tinystories"

    batch_size:int=16
    learning_rate:float=3e-4
    min_lr:float=1e-5
    weight_decay:float = 0.1
    beta1:float=0.9
    beta2:float=0.95
    grad_clip:float=1.0
    gradient_accumulation_steps:int = 8

    warmup_steps:int=200
    max_iters:int=100000
    eval_interval:int=500#calculate val loss for every 500 trainig steps.
    eval_iters:int=100#calculate val loss for 100 steps.
    save_interval:int=10000#saves the model weights every 10,000 steps.
    log_interval:int=500
    block_size:int=256

    device:str="cuda"
    dtype:str="bfloat16"
    compile:bool=False

    out_dir:str="checkpoints_gqa_rope"