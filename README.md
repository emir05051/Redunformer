# Redunformer

## Head-wise redundancy measurement

The first measurement captures each attention head immediately before the
layer's output projection. It computes an absolute centered-cosine similarity
matrix between head activation traces. A head's redundancy score is the maximum
similarity to any other head in the same layer (0 = distinct, 1 = identical or
sign-flipped). The same run records per-head activation mean, standard deviation,
RMS, and near-zero fraction.

```bash
uv run python scripts/measure_redundancy.py --model gpt2
```

Outputs are written to `configs/experiments/redundancy/`:

- `*_head_redundancy.json`: complete matrices, scores, and activation statistics
- `*_head_redundancy.csv`: sortable layer/head table
- `*_head_redundancy.png`: layer-by-head score heatmap and the most-redundant
  layer's similarity matrix

Use `--samples`, `--max-length`, and `--max-tokens` to control calibration cost.

Qwen3.5 hybrid models interleave full attention with gated-delta linear
attention. This implementation measures/prunes standard attention heads only
(layers 3, 7, 11, ... in Qwen3.5-4B); linear-attention value heads are skipped
because they have different semantics and dimensions.

## C4 loss recovery

Recover loss after pruning by training a LoRA adapter on a deterministic packed
subset of C4. WikiText is used only for the unpruned, pruned, and recovered
evaluations.

```bash
uv run python scripts/run_recovery.py --model gpt2 --pruning-method gradient --ratio 0.2
uv run python scripts/run_recovery.py --model Qwen/Qwen3.5-4B --pruning-method gradient --ratio 0.2
uv run python scripts/run_recovery.py --model meta-llama/Llama-3.2-3B --pruning-method gradient --ratio 0.2
```

GPT-2 uses bf16 LoRA. Qwen3.5-4B and Llama-3.2-3B use 4-bit NF4 QLoRA
when `--quantization auto` is selected. Add `--run-lm-harness` to evaluate the
existing downstream task suite before and after recovery. Run artifacts are
written under `outputs/recovery/`, with a compact metrics copy under
`configs/experiments/<model>/`.
