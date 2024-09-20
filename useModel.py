import torch
import tiktoken
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.2

class GPT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = torch.nn.ModuleDict(dict(
            wte = torch.nn.Embedding(config.vocab_size, config.n_embd),
            wpe = torch.nn.Embedding(config.block_size, config.n_embd),
            h = torch.nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = torch.nn.LayerNorm(config.n_embd)
        ))

        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

class Block(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(config.n_embd)
        self.ln2 = torch.nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CausalSelfAttention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = torch.nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = torch.nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = torch.nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = torch.nn.GELU()
        self.c_proj = torch.nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

def generate_text(model, prompt, device, num_return_sequences=4, max_length=32, device_type='cuda', seed=42):
    model.eval()
    enc = tiktoken.get_encoding("gpt2")
    
    tokens = enc.encode(prompt)
    print(f"Tokenized prompt: {tokens}")
    
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    xgen = tokens.to(device)
    
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(seed)
    
    while xgen.size(1) < max_length:
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(xgen)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)
    
    generated_sequences = []
    for i in range(num_return_sequences):
        tokens = xgen[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        generated_sequences.append(decoded)
        print(f"Sample {i}: {decoded}")
    
    return generated_sequences

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    config = GPTConfig(**checkpoint['config'].__dict__)
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    
    print(f"Model config: {config}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model

if __name__ == "__main__":
    model_path = "model_40000.pt"
    model = load_model(model_path)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    device = torch.device(device)
    model.to(device)
    
    print(f"Using device: {device}")
    
    prompt = input("Enter your prompt: ")
    
    # Generate text with different seeds
    seeds = [42, 100, 200, 300]
    for seed in seeds:
        print(f"\nGenerating with seed={seed}")
        generated_sequences = generate_text(model, prompt, device, num_return_sequences=4, max_length=100, device_type=device_type, seed=seed)
        print("-" * 50)