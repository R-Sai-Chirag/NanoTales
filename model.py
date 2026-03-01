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
        self.register_buffer("freqs_cis", torch.polar(torch.ones_like(freqs), freqs))  # ← fix 2

    def forward(self, x):
        # x shape: [batch, n_head, seq_len, head_dim]
        seq_len = x.shape[2]
        x_complex = torch.view_as_complex(                   
            x.float().reshape(*x.shape[:-1], -1, 2)
        )
        freqs_cis = self.freqs_cis[:seq_len, :].unsqueeze(0).unsqueeze(0)  # [1,1,seq_len,head_dim//2]
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
    
class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc=nn.Linear(config.d_model,4*config.d_model,bias=config.bias)
        self.c_proj=nn.Linear(4*config.d_model,config.d_model,bias=config.bias)
        self.act=nn.GELU()
        self.dropout=nn.Dropout(config.dropout)

    def forward(self,x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))
    
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln1=RMSNorm(config.d_model)
        self.att=MultiHeadAttention(config)
        self.mlp=MLP(config)
        self.ln2=RMSNorm(config.d_model)
    
    def forward(self, x):
        x = x + self.att(self.ln1(x))   
        x = x + self.mlp(self.ln2(x))   
        return x
    
class GPT(nn.Module):
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
        self.transformer.wte.weight=self.lm_head.weight

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

    def forward(self,idx,targets=None):
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

        for block in self.transformer.h:
            x=block(x)

        x=self.transformer.ln_f(x)

        if targets is not None:
            logits=self.lm_head(x)
            loss=F.cross_entropy(
                logits.view(-1,logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            return logits,loss
        else:
            logits=self.lm_head(x[:,[-1],:])
            return logits,None
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits     = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx        = torch.cat([idx, next_token], dim=1)
        return idx
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())