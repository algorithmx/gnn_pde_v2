# Research Papers on Adaptive Temperature in Attention Mechanisms

This directory contains 4 key research papers (converted to text) that provide
precise references for implementing adaptive/learnable temperature in 
physics-informed attention mechanisms.

## Papers Summary

### 1. Transolver++ (ICML 2025) - `transolver_plus_plus.txt`
**Citation:** Luo et al., "Transolver++: An Accurate Neural Solver for PDEs on Million-Scale Geometries," ICML 2025.

**Key Contribution:** Introduces "Ada-Temp" (Adaptive Temperature) for physics-attention:
- Local adaptive temperature: `T_i = τ_0 + Linear_T(h_i)` computed per mesh point
- Prevents state homogenization on large meshes by dynamically sharpening/softening slice weight distributions  
- Combined with Gumbel-Softmax reparameterization for differentiable sampling
- Formula: `Rep-Slice(x, τ) = Softmax((Linear(x) - log(-log ε)) / τ)`

**Relevant Sections:** Lines 260-270, 385-420 in the text file.

---

### 2. Query-Key Normalization for Transformers (EMNLP 2020) - `qk_norm.txt`
**Citation:** Henry et al., "Query-Key Normalization for Transformers," Findings of EMNLP 2020.

**Key Contribution:** QK-Norm with learnable temperature:
- Per-head learnable scalar value for attention scores after L2 normalization
- Eliminates traditional `sqrt(head_dimension)` scaling factor
- Formula: `softmax(g * Q_hat * K_hat^T) * V` where g is learnable
- g initialized as: `g_0 = log_2(L^2 - L)` where L is 97.5th percentile sequence length
- Improves low-resource translation by +0.928 BLEU on average

**Relevant Sections:** Lines 145-160, 265-310 in the text file.

---

### 3. Low-Width Approximations for Graph Transformers (NeurIPS 2023 Workshop) - `low_width_graph_transformers.txt`
**Citation:** Shirzad et al., "Low-Width Approximations and Sparsification for Scaling Graph Transformers," NeurIPS GLFrontiers 2023.

**Key Contribution:** Temperature annealing schedule:
- Start with τ=1.0, gradually anneal to 0.05 by end of training
- Initial phase of c epochs with τ=1 to learn important neighbors
- Update formula: `τ_epoch = max(f^(epoch-c), 0.05)` where f ∈ [0.95, 0.98]
- For fast-converging models: c=5, f=0.98
- For slower-converging models: c=10, f=0.95

**Relevant Sections:** Lines 240-260 in the text file.

---

## Implementation Recommendations Summary

Based on these papers, here are the recommended approaches for implementing 
adaptive temperature in physics-informed attention:

### Option 1: Simple Learnable Temperature (from QK-Norm)
```python
self.temperature = nn.Parameter(torch.ones(1))
# In forward:
scores = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
```

### Option 2: Per-Point Adaptive Temperature (from Transolver++)
```python
self.temp_proj = nn.Linear(hidden_dim, 1)
self.tau_0 = 1.0  # temperature constant
# In forward:
tau = self.tau_0 + self.temp_proj(x)  # per-point temperature
slice_weights = F.softmax(slice_logits / tau, dim=-1)
```

### Option 3: Temperature Annealing (from Low-Width Graph Transformers)
```python
# During training, update temperature each epoch:
self.temperature = max(0.98 ** (epoch - c), 0.05)
# where c is initial phase (5 or 10 epochs)
```

### Option 4: Gumbel-Softmax with Adaptive Temperature (from Transolver++)
```python
epsilon = torch.rand_like(slice_logits)
gumbel_noise = -torch.log(-torch.log(epsilon))
slice_weights = F.softmax((slice_logits - gumbel_noise) / tau, dim=-1)
```

### Important Constraints:
- Apply `softplus` or `clamp(min=0.1)` to prevent temperature collapse to zero
- Initialize near τ=1.0 for stable training
- Consider per-head vs per-layer vs per-point temperature based on problem scale

---

## File Sizes
- `transolver_plus_plus.pdf`: ~20.4 MB (arXiv:2502.02414)
- `qk_norm.pdf`: ~589 KB (ACL Anthology)
- `gnot.pdf`: ~4.2 MB (PMLR Proceedings)
- `low_width_graph_transformers.pdf`: ~511 KB (OpenReview)

## Text Extraction Date
March 16, 2026
