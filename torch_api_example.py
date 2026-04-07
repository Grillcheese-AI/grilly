"""VSA-LM v3b: Sequence-mean LiquidCell, batched.

Reverts to the proven v1 architecture (one LiquidCell step per sequence via
x_mean), but fully batched on GPU with B=64. Loads vsa_lm_v1_resume.pt and
continues cosine decay from wherever the checkpoint left off.

Why: per-token recurrence (v3) hit 0.2 stp/s on A100 due to Python loop
overhead. Sequence-mean is O(1) per layer per sequence — GPU-friendly and
the architecture that trained the checkpoint in the first place.
"""

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQ_LEN = 256
DATA_DIR = 'vsa_lm_data'

# ── Data ──
tokens = np.load(f'{DATA_DIR}/tokens.npy')
vocab = int(np.load(f'{DATA_DIR}/vocab.npy')[0])
n = len(tokens)
tr, va = tokens[:int(.8 * n)], tokens[int(.8 * n):int(.9 * n)]


def mkseqs(t, sl):
    x, y = [], []
    for i in range(0, len(t) - sl - 1, sl // 2):
        x.append(t[i:i + sl])
        y.append(t[i + 1:i + sl + 1])
    return (torch.tensor(np.array(x), dtype=torch.long),
            torch.tensor(np.array(y), dtype=torch.long))


train_x, train_y = mkseqs(tr, SEQ_LEN)
val_x, val_y = mkseqs(va, SEQ_LEN)
print(f'Vocab={vocab}, Train={len(train_x)}, Val={len(val_x)}')


def compute_ppl(model, x_data, y_data, max_samples=100):
    model.eval()
    total_loss, n_tok = 0.0, 0
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        # Batched eval for speed
        bs = 32
        for i in range(0, min(len(x_data), max_samples), bs):
            xb = x_data[i:i + bs].to(device)
            yb = y_data[i:i + bs].to(device)
            logits = model(xb)  # (B, S, V)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                yb.reshape(-1),
                reduction='sum',
            )
            total_loss += loss.item()
            n_tok += yb.numel()
    model.train()
    return math.exp(min(total_loss / max(n_tok, 1), 20))


# ── Config ──
D_MODEL = 384
N_LAYERS = 12
D_FFN = 1152
LR = 1e-4
TRAIN_STEPS = 100000
VAL_EVERY = 200
GRAD_CLIP = 1.0
BATCH_SIZE = 16
CAPSULE_DIM = 32
USE_AMP = True  # fp16 autocast — ~1.5-2x speedup on cdist


# ── Model ──
class AdditionLinearCUDA(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in).uniform_(-0.1, 0.1))
        self.bias = nn.Parameter(torch.zeros(d_out))
        self.d_in = d_in

    def forward(self, x):
        # x: (..., d_in) → (..., d_out). Flatten leading dims for cdist.
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        dist = torch.cdist(x_flat.unsqueeze(0), self.weight.unsqueeze(0), p=1).squeeze(0)
        out = -dist + self.bias
        return out.reshape(*orig_shape[:-1], -1)


class LiquidCellCUDA(nn.Module):
    """Sequence-mean LiquidCell — one step per sequence (not per token).

    For batched input (B, d), processes all B sequences in parallel in a
    single forward pass. Matches the v1 architecture that trained
    vsa_lm_v1_resume.pt.
    """

    def __init__(self, d, dt=0.02, tau_min=0.02, tau_max=2.0):
        super().__init__()
        s = math.sqrt(2.0 / (d + d))
        self.W = nn.Parameter(torch.randn(d, d) * s)
        self.U = nn.Parameter(torch.randn(d, d) * s)
        self.b = nn.Parameter(torch.zeros(d))
        self.V = nn.Parameter(torch.randn(d, d) * s)
        self.c = nn.Parameter(torch.randn(d) * 0.1)
        self.register_buffer('h', torch.zeros(d))
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max

    def step(self, x):
        # Single sequence: x is (d,)
        tau = self.tau_min + F.softplus(self.V @ x + self.c)
        tau = torch.clamp(tau, max=self.tau_max)
        a = torch.tanh(self.W @ self.h + self.U @ x + self.b)
        dh = -self.h / tau.clamp(min=1e-6) + a
        self.h = (self.h + self.dt * dh).detach()
        return self.h

    def step_batched(self, x):
        # x: (B, d). Broadcasts h across batch, single parallel update.
        B = x.shape[0]
        tau = self.tau_min + F.softplus(x @ self.V.T + self.c)  # (B, d)
        tau = torch.clamp(tau, max=self.tau_max)
        h_b = self.h.unsqueeze(0).expand(B, -1)  # (B, d) — shared state
        a = torch.tanh(h_b @ self.W.T + x @ self.U.T + self.b)  # (B, d)
        dh = -h_b / tau.clamp(min=1e-6) + a
        new_h = h_b + self.dt * dh  # (B, d)
        # Update the shared buffer to the batch mean (detached for next call)
        self.h = new_h.mean(dim=0).detach()
        return new_h

    def reset(self):
        self.h.zero_()


class VSALayerCUDA(nn.Module):
    def __init__(self, d, d_ffn):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.ffn_up = AdditionLinearCUDA(d, d_ffn)
        self.ffn_down = AdditionLinearCUDA(d_ffn, d)
        self.liquid = LiquidCellCUDA(d)
        self.d = d

    def forward(self, x, plasticity=0.5, consolidation=0.5):
        if x.ndim == 3:
            return self._forward_batch(x, plasticity, consolidation)
        return self._forward_single(x, plasticity, consolidation)

    def _forward_single(self, x, plasticity, consolidation):
        # x: (S, d)
        h = self.ln(x)
        x_mean = h.mean(dim=0)  # (d,)
        self.liquid.dt = 0.01 + 0.03 * plasticity
        self.liquid.tau_min = 0.02 + 0.08 * consolidation
        temporal = self.liquid.step(x_mean)  # (d,)

        h_up = torch.sign(self.ffn_up(h) / math.sqrt(self.ffn_up.d_in))
        h_ffn = self.ffn_down(h_up) / math.sqrt(self.ffn_down.d_in)

        gate = torch.tanh(temporal)  # (d,)
        y = (0.5 + plasticity) * h_ffn * (1.0 + 0.1 * gate)
        return x + y

    def _forward_batch(self, x, plasticity, consolidation):
        # x: (B, S, d)
        h = self.ln(x)
        x_mean = h.mean(dim=1)  # (B, d)
        self.liquid.dt = 0.01 + 0.03 * plasticity
        self.liquid.tau_min = 0.02 + 0.08 * consolidation
        temporal = self.liquid.step_batched(x_mean)  # (B, d)

        h_up = torch.sign(self.ffn_up(h) / math.sqrt(self.ffn_up.d_in))
        h_ffn = self.ffn_down(h_up) / math.sqrt(self.ffn_down.d_in)

        gate = torch.tanh(temporal).unsqueeze(1)  # (B, 1, d) — broadcasts over S
        y = (0.5 + plasticity) * h_ffn * (1.0 + 0.1 * gate)
        return x + y


class VSALMModel(nn.Module):
    def __init__(self, vocab, d, d_ffn, n_layers, max_seq):
        super().__init__()
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        self.capsule_embed = nn.Embedding(vocab, CAPSULE_DIM)
        self.capsule_proj = nn.Linear(CAPSULE_DIM, d, bias=False)
        self.pos = nn.Parameter(torch.randn(max_seq + 16, d) * 0.02)
        self.out_proj = nn.Linear(d, vocab, bias=False)
        self.layers = nn.ModuleList([VSALayerCUDA(d, d_ffn) for _ in range(n_layers)])
        self.scale = math.sqrt(d)
        nn.init.normal_(self.embed.weight, 0, 0.02)
        nn.init.normal_(self.capsule_embed.weight, 0, 0.01)

    def forward(self, ids):
        if ids.ndim == 1:
            S = ids.shape[0]
            x = self.embed(ids) + self.pos[:S]
            caps = self.capsule_embed(ids)
            x = x + self.capsule_proj(caps)
            with torch.no_grad():
                cm = caps.mean(dim=0)
                plasticity = cm[14].clamp(0, 1).item()
                consolidation = cm[18].clamp(0, 1).item()
            for layer in self.layers:
                x = layer(x, plasticity=plasticity, consolidation=consolidation)
            return self.out_proj(x / self.scale)

        # Batched: (B, S)
        B, S = ids.shape
        x = self.embed(ids) + self.pos[:S].unsqueeze(0)
        caps = self.capsule_embed(ids)
        x = x + self.capsule_proj(caps)
        with torch.no_grad():
            cm = caps.mean(dim=(0, 1))
            plasticity = cm[14].clamp(0, 1).item()
            consolidation = cm[18].clamp(0, 1).item()
        for layer in self.layers:
            x = layer(x, plasticity=plasticity, consolidation=consolidation)
        return self.out_proj(x / self.scale)

    def reset_liquid(self):
        for layer in self.layers:
            layer.liquid.reset()


# ── Load checkpoint ──
model = VSALMModel(vocab, D_MODEL, D_FFN, N_LAYERS, SEQ_LEN).to(device)

ckpt_path = 'vsa_lm_v1_resume.pt' if os.path.exists('vsa_lm_v1_resume.pt') else 'vsa_lm_best.pt'
checkpoint = torch.load(ckpt_path, map_location=device)
if 'model' in checkpoint:
    model.load_state_dict(checkpoint['model'], strict=True)
    ckpt_step = checkpoint.get('step', 0)
    ckpt_ppl = checkpoint.get('best_ppl', 0)
    print(f'Loaded {ckpt_path} (step={ckpt_step}, PPL={ckpt_ppl:.1f})')
else:
    model.load_state_dict(checkpoint, strict=True)
    ckpt_step = 0
    print(f'Loaded {ckpt_path}')

n_params = sum(p.numel() for p in model.parameters())
print(f'VSA-LM v3b: d={D_MODEL}, L={N_LAYERS}, B={BATCH_SIZE}, params={n_params/1e6:.1f}M')

with torch.no_grad():
    ppl = compute_ppl(model, val_x, val_y)
    print(f'Checkpoint PPL: {ppl:.1f}')

# ── Train ──
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
# Cosine over remaining steps
remaining = max(TRAIN_STEPS - ckpt_step, 1000)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, remaining, eta_min=1e-6)
scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

t0 = time.time()
best_ppl = ppl
step = 0

while step < remaining:
    perm = torch.randperm(len(train_x))
    for i in range(0, len(perm) - BATCH_SIZE, BATCH_SIZE):
        if step >= remaining:
            break

        ids_batch = train_x[perm[i:i + BATCH_SIZE]].to(device)
        labels_batch = train_y[perm[i:i + BATCH_SIZE]].to(device)

        model.reset_liquid()
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=USE_AMP, dtype=torch.float16):
            logits = model(ids_batch)  # (B, S, V)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels_batch.reshape(-1),
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % VAL_EVERY == 0:
            ppl = compute_ppl(model, val_x, val_y)
            el = time.time() - t0
            sps = (step + 1) / el if el > 0 else 0
            lr_now = scheduler.get_last_lr()[0]
            print(f'step={step:6d} lr={lr_now:.5f} | loss={loss.item():.3f} | '
                  f'PPL={ppl:.1f} | {sps:.1f} stp/s (B={BATCH_SIZE})')
            if ppl < best_ppl:
                best_ppl = ppl
                torch.save(
                    {'model': model.state_dict(), 'step': ckpt_step + step, 'best_ppl': best_ppl},
                    'vsa_lm_v3b_best.pt',
                )

        step += 1

print(f'\nDone: {step} steps, best PPL={best_ppl:.1f}, time={time.time()-t0:.0f}s')
