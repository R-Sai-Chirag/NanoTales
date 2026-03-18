import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from config import GPTConfig

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight
    
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        head_dim = config.d_model // config.n_head         
        theta = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(0, config.block_size)
        freqs = positions.unsqueeze(1) * theta.unsqueeze(0)
        self.register_buffer("freqs_cis", torch.polar(torch.ones_like(freqs), freqs))

    def forward(self, x,curr_pos=0):
        # x shape: [batch, n_head, seq_len, head_dim]
        seq_len = x.shape[2]
        x_complex = torch.view_as_complex(                   
            x.float().reshape(*x.shape[:-1], -1, 2)
        )
        freqs_cis = self.freqs_cis[curr_pos:curr_pos+seq_len, :].unsqueeze(0).unsqueeze(0)  # [1,1,seq_len,head_dim//2]
        x_rotated = x_complex * freqs_cis
        x_rotated = torch.view_as_real(x_rotated).flatten(3)
        return x_rotated.type_as(x)

    
class MultiHeadAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.d_model%config.n_head==0

        self.n_head=config.n_head
        self.d_model=config.d_model
        self.head_dim=config.d_model//config.n_head
        self.dropout=nn.Dropout(config.dropout)
        self.bias=config.bias
        self.c_attn=nn.Linear(self.d_model,self.d_model*3,bias=self.bias)
        self.c_proj=nn.Linear(self.d_model,self.d_model,bias=self.bias)
        self.rope=RotaryPositionalEncoding(config) if config.use_rope else None

        self.flash=hasattr(F,"scaled_dot_product_attention")

        if not self.flash:
            self.register_buffer("mask",torch.tril(torch.ones(config.block_size,config.block_size))
                                 .view(1, 1, config.block_size, config.block_size))

    def forward(self,x):
        B,T,C=x.size()

        q,k,v=self.c_attn(x).split(self.d_model,dim=-1)

        q=q.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        k=k.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        v=v.view(B,T,self.n_head,self.head_dim).transpose(1,2)

        if self.rope is not None:
            q=self.rope(q)
            k=self.rope(k)

        if self.flash:
            y=F.scaled_dot_product_attention(q,k,v,
                                             attn_mask=None,
                                             dropout_p=self.dropout.p if self.training else 0,
                                             is_causal=True)
        else:
            att=(q@k.transpose(2,3))*(1.0/math.sqrt(self.head_dim))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
            att_weights=F.softmax(att,dim=-1)
            att_weights=self.dropout(att_weights)
            y=att_weights@v

        y=y.transpose(1,2).contiguous().view(B,T,C)
        y=self.dropout(self.c_proj(y))
        return y
    
class GroupQueryAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.d_model=config.d_model
        self.n_head=config.n_head
        self.n_groups=config.n_groups
        self.rope=RotaryPositionalEncoding(config) if config.use_rope else None
        assert config.d_model%config.n_head==0
        assert config.n_head%config.n_groups==0

        self.heads_per_group=config.n_head//config.n_groups
        self.head_dim=config.d_model//config.n_head
        self.kv_dim=config.n_groups*self.head_dim

        self.c_attn=nn.Linear(config.d_model,config.d_model+2*self.kv_dim,bias=config.bias)
        self.c_proj=nn.Linear(config.d_model,config.d_model,bias=config.bias)
        self.dropout=nn.Dropout(config.dropout)

        self.flash=hasattr(F,"scaled_dot_product_attention")

        if not self.flash:
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self,x,past_kv=None,use_cache=True):
        B,T,C=x.size()

        curr_pos=past_kv[0].shape[2] if past_kv is not None else 0

        q,k,v=self.c_attn(x).split([self.config.d_model,
                                    self.kv_dim,
                                    self.kv_dim],dim=-1)

        q=q.view(B,T,self.n_head,self.head_dim).transpose(1,2)
        k=k.view(B,T,self.config.n_groups,self.head_dim).transpose(1,2)
        v=v.view(B,T,self.config.n_groups,self.head_dim).transpose(1,2)

        if self.rope:
            q=self.rope(q,curr_pos)
            k=self.rope(k,curr_pos)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None
        
        k=k.repeat_interleave(self.heads_per_group,dim=1)
        v=v.repeat_interleave(self.heads_per_group,dim=1)

        if self.flash:
            y=F.scaled_dot_product_attention(q,k,v,
                                             dropout_p=self.dropout.p if self.training else 0,
                                             attn_mask=None,
                                             is_causal=True if T>1 else False)
            
        else:
            attn_scores=(q@k.transpose(2,3))*(1.0/math.sqrt(self.head_dim))
            if T > 1:
                full_len = k.shape[2]
                attn_scores = attn_scores.masked_fill(
                    self.mask[:, :, :T, :full_len] == 0, float("-inf")
                )
            attn_weights=F.softmax(attn_scores,dim=-1)
            attn_weights=self.dropout(attn_weights)
            y=attn_weights@v

        y=y.transpose(1,2).contiguous().view(B,T,C)
        y=self.dropout(self.c_proj(y))

        return y,present_kv
    
    def extra_repr(self)->str:
        mode=("MHA" if self.config.n_groups==self.config.n_head 
              else "MQA" if self.config.n_groups==1 else "GQA")
        
        return (f"mode={mode}, n_head={self.n_head}, n_groups={self.n_groups}, "
                f"head_dim={self.head_dim}, heads_per_group={self.heads_per_group}, "
                f"d_model={self.d_model}, kv_dim={self.kv_dim}")
    


class SwiGLU(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.w_up=nn.Linear(config.d_model,(config.d_model*8)//3,bias=config.bias)
        self.w_gate=nn.Linear(config.d_model,(config.d_model*8)//3,bias=config.bias)
        self.w_down=nn.Linear((config.d_model*8)//3,config.d_model,bias=config.bias)
        self.silu=nn.SiLU()

    def forward(self,x):
        activations=self.silu(self.w_gate(x))
        representations=self.w_up(x)
        return self.w_down(activations*representations)
    
    
    
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln1=RMSNorm(config.d_model)
        self.att=GroupQueryAttention(config) if config.use_gqa else MultiHeadAttention(config)
        self.mlp=SwiGLU(config)
        self.ln2=RMSNorm(config.d_model)
    
    def forward(self, x,past_kv=None,use_cache=False):
        if isinstance(self.att, GroupQueryAttention):
            attn_out, present_kv = self.att(self.ln1(x), past_kv=past_kv, use_cache=use_cache)
        else:
            attn_out = self.att(self.ln1(x))  # MHA — no cache support
            present_kv = None
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))   
        return x,present_kv
    
class GPT2(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.d_model),
            dropout=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config=config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.d_model)
        ))
        self.wpe=nn.Embedding(config.block_size,config.d_model) if not config.use_rope else None

        self.lm_head=nn.Linear(config.d_model,config.vocab_size,bias=False)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

        #initializing c_proj weights with less standard deviation.
        for pn,p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p,mean=0.0,std=0.02/math.sqrt(2*config.n_layer))

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module,nn.Embedding):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)

    def forward(self,idx,targets=None,use_cache=False,past_kvs=None):
        device=idx.device
        b,t=idx.size()

        assert t<=self.config.block_size,f"Sequence {t} > block_size {self.config.block_size}"

        if not self.config.use_rope:
            pos=torch.arange(0,t,dtype=torch.long,device=device)
            tok_emb=self.transformer.wte(idx)
            pos_emb=self.wpe(pos)
            x=self.transformer.dropout(tok_emb+pos_emb)
        
        else:
            tok_emb=self.transformer.wte(idx)
            x=self.transformer.dropout(tok_emb)

        present_kvs=[]
        for i,block in enumerate(self.transformer.h):
            past_kv=past_kvs[i] if past_kvs is not None else None
            x, present_kv=block(x,use_cache=use_cache,past_kv=past_kv)
            present_kvs.append(present_kv)


        x=self.transformer.ln_f(x)

        if targets is not None:
            logits=self.lm_head(x)
            loss=F.cross_entropy(
                logits.view(-1,logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            return logits,loss,None
        else:
            logits=self.lm_head(x[:,[-1],:])
            return logits,None,present_kvs if use_cache else None
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_kv_cache=True):
        past_kvs = None
        current_input = idx

        for step in range(max_new_tokens):
            if use_kv_cache:
                logits, _, past_kvs = self(current_input, use_cache=True, past_kvs=past_kvs)
            else:
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                logits, _, _ = self(idx_cond, use_cache=False, past_kvs=None)  # ← full sequence every step

            if step % 50 == 0:
                if past_kvs is not None and past_kvs[0] is not None:
                    cache_len = past_kvs[0][0].shape[2]
                    print(f"[Cache] step {step} | cache_len={cache_len} | input_len={current_input.shape[1]}")

            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
            current_input = next_token

        return idx
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())