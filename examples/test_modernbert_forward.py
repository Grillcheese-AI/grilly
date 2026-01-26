"""Test ModernBERT forward pass order"""

from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('ibm-granite/granite-embedding-english-r2')
auto_model = model._modules['0'].auto_model
layer0 = auto_model.layers[0]

x = torch.randn(1, 7, 768)
print('Input:', x.shape, x.mean().item(), x.std().item())

# ModernBERT forward:
# 1. Attention (no pre-norm)
attn_out = layer0.attn(x)
print('After attn:', attn_out.shape, attn_out.mean().item(), attn_out.std().item())

# 2. Residual (no norm after attention!)
if hasattr(layer0, 'attn_norm'):
    if isinstance(layer0.attn_norm, torch.nn.Identity):
        attn_residual = attn_out + x  # No norm, just residual
        print('After attn residual (no norm):', attn_residual.shape, attn_residual.mean().item(), attn_residual.std().item())
    else:
        attn_residual = layer0.attn_norm(attn_out + x)
        print('After attn_norm:', attn_residual.shape, attn_residual.mean().item(), attn_residual.std().item())
else:
    attn_residual = attn_out + x
    print('After attn residual (no norm):', attn_residual.shape, attn_residual.mean().item(), attn_residual.std().item())

# 3. MLP
mlp_out = layer0.mlp(attn_residual)
print('After mlp:', mlp_out.shape, mlp_out.mean().item(), mlp_out.std().item())

# 4. Residual + LayerNorm
if hasattr(layer0, 'mlp_norm'):
    final_out = layer0.mlp_norm(mlp_out + attn_residual)
    print('After mlp_norm (with residual):', final_out.shape, final_out.mean().item(), final_out.std().item())
else:
    final_out = mlp_out + attn_residual
    print('After mlp residual (no norm):', final_out.shape)

print('\nSummary:')
print('  attn_norm is Identity:', isinstance(layer0.attn_norm, torch.nn.Identity) if hasattr(layer0, 'attn_norm') else 'N/A')
print('  mlp_norm is LayerNorm:', isinstance(layer0.mlp_norm, torch.nn.LayerNorm) if hasattr(layer0, 'mlp_norm') else 'N/A')
