import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
from transformers import GPT2LMHeadModel, set_seed
import tiktoken
import os
from os import path as osp


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # register_buffer for temporary var: an lower triangular matrix for masking
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))
        # print(self.bias[:,:,:10,:10]); exit(0)

    def forward(self, x): # B,T,C
        B,T,C = x.shape
        qkv = self.c_attn(x)
        q,k,v = qkv.split(self.n_embd, dim=2) # split at dim 2 with length self.n_emb
        q = q.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # B,T,nh,hs -> B,nh,T,hs
        k = k.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # B,T,nh,hs -> B,nh,T,hs
        v = v.view(B, T, self.n_head, C//self.n_head).transpose(1, 2) # B,T,nh,hs -> B,nh,T,hs
        attn = q @ k.transpose(-2, -1) * (1./math.sqrt(k.shape[-1])) # B,nh,T,T
        attn = attn.masked_fill(self.bias[:,:,:T,:T]==0, float('-inf'))
        attn = F.softmax(attn, dim=-1) 
        x = attn @ v # B,nh,T,hs
        x = x.transpose(1, 2).contiguous().view(B, T, C)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)        

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embdding dimension


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList(Block(config) for _ in range(config.n_head)),
            ln_f = nn.LayerNorm(config.n_embd),  # final_ln
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # bias=False ???

    def forward(self, idx, targets=None): # idx is the tokenized result, (B,T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Can't forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_embd = self.transformer.wpe(pos)
        token_embd = self.transformer.wte(idx)
        x = pos_embd + token_embd # B, T, C
        for layer in self.transformer.h:
            # print('------')
            x = layer(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # B, T, vocab_size
        return logits
    
    @classmethod
    def from_pretrained(cls, model_type):
        "Loads pretrained GPT-2 model from huggingface transformers"
        assert model_type in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']
        print(f"loading weights from pretrained gpt: {model_type}")

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024), # 350M
            'gpt2-large':  dict(n_layer=36, n_head=20, n_embd=1280), # 774M
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600), # 1558M
        }[model_type]
        config = GPTConfig(**config_args)
        model = GPT(config)
        
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')] # ignore the buffer params
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']\
        # nn.Linear(x,y): output = input * weight^T + bias, so shape of weight is (y,x)
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # print(k, sd_hf[k].shape[::-1], sd[k].shape)
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GPT.from_pretrained('gpt2')
# model = GPT(GPTConfig())
model.to(device)
model.eval()

tokenizer = tiktoken.get_encoding('gpt2')

def test_model():
    B = 5
    max_length = 30

    tokenizer = tiktoken.get_encoding('gpt2')
    input_sentence = "Hello, I'm a language model,"
    x = tokenizer.encode(input_sentence)
    x = torch.tensor(x, dtype=torch.long, device=device)
    x = x.unsqueeze(0).repeat(B, 1) # B, T

    set_seed(42)
    while x.shape[1] < max_length:
        with torch.no_grad():
            logits = model(x) # B, T, vocab_size
            logits = logits[:, -1, :]  # B, vocab_size
            probs = F.softmax(logits, dim=-1)  # logits != probs , softmax is needed
            # 50 is default in huggingface pipeline.  Clamp down rare probs.
            topk_probs, topk_ids = torch.topk(probs, 50, dim=-1) # (B, 50). 
            ix = torch.multinomial(topk_probs, 1) # sampling
            nx = torch.gather(topk_ids, -1, ix) # get data a/to dim and idx
            x = torch.cat((x, nx), dim=1) # B, T+1

    lst = x.tolist()
    print(input_sentence)
    for idx in lst:
        print(">", tokenizer.decode(idx))

test_model()