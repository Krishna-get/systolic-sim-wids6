# Layer topologies

CSV files in **SCALE-Sim topology format**, so the same file feeds your
simulator and SCALE-Sim when you validate in Week 4.

## Columns

```
Layer name, IFMAP Height, IFMAP Width, Filter Height, Filter Width,
            Channels, Num Filter, Strides
```

Note the **trailing comma** on every row. SCALE-Sim's parser expects it;
don't strip it.

## Convolution layers

Read straight off. `resnet18.csv` is a full ResNet-18: 20 conv layers plus
the final fully-connected layer, including the 1×1 downsample convolutions
in the residual branches (`*_down`). Spatial dimensions are post-pooling
where relevant.

## GEMM layers — the 1×1 trick

Transformers have no convolutions, so a GEMM of shape `M×K @ K×N` is
encoded as a 1×1 convolution:

| GEMM | CSV column |
|---|---|
| M (rows of the activation matrix) | IFMAP Height |
| — | IFMAP Width = 1 |
| — | Filter Height = Filter Width = 1 |
| K (reduction dimension) | Channels |
| N (output columns) | Num Filter |

Your `im2col()` from Week 2 should return `(M, K, N)` unchanged for these,
since a 1×1 filter with stride 1 duplicates nothing.

## What's in the transformer files

`bert_base.csv` and `vit_base.csv` list **one transformer block**, not the
whole network. Both models stack 12 identical blocks, so multiply your
per-block results by 12 and add the embedding/head layers (which are listed
separately: `pooler`, `patch_embed`, `head`).

Doing it this way keeps the sweeps in Weeks 3 and 4 down from ~1500
simulations to ~130 without losing anything — the blocks are identical, so
simulating all twelve tells you nothing the first one didn't.

Assumptions baked in:
- **BERT-base:** sequence length 128, hidden 768, intermediate 3072, 12 heads
- **ViT-base:** 224×224 input, patch 16 → 196 patches + 1 class token = 197
  tokens, hidden 768, MLP 3072, 12 heads

The attention `scores` and `context` layers are the per-head batched
matmuls, flattened across heads into the M dimension (12 heads × 128 tokens
= 1536 for BERT; 12 × 197 = 2364 for ViT). This is a simplification — real
hardware handles batched matmul differently — and it's worth one sentence in
your Week 3 limitations section.

## Adding your own

Any network you like. Keep the format identical so SCALE-Sim can read it,
and keep layer names unique so your results CSVs are joinable.
