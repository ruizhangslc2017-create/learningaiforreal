# Attention

Goal: explain how token vectors produce attention scores, normalized weights,
and output vectors; then explain why causal masking blocks future information.

## Use this learning unit

1. Read the three-token example in [notes.md](notes.md).
2. Predict the first causal output before running [implementation.py](implementation.py).
3. Run the program and compare both modes:

   ```bash
   python3 implementation.py
   ```

4. Open [experiments.ipynb](experiments.ipynb) in a Jupyter-compatible editor.
   Run from the attention folder or the AI root. Its calculations use the
   standard library; optional plots appear if Matplotlib is installed.
5. Read the focused [paper notes](paper-notes.md), then return to the equations.

## Current state

- Stable row-wise softmax and an inspectable attention calculation.
- An optional causal mask applied before softmax.
- A projection helper for fixed `W_Q`, `W_K`, and `W_V` matrices.
- A worked numerical example and a future-token isolation check.
- Four notebook experiments with saved outputs: masking, changing values,
  scaling, and separate projections.

The token vectors and projection weights are hand-chosen. There is no training
loop yet. The causal mask uses equal-length, aligned sequences; cached decoding
and cross-attention are later extensions.

## Check your understanding

- Identify what row `i`, column `j` means in the weight matrix.
- Explain why a row sums to one, and why columns need not.
- Show why the first causal output equals the first value vector.
- Change a value without changing its key; predict what happens to the weights.
- Rebuild the core attention loop without looking at the implementation.

## Next step

Implement two heads with separate projections, concatenate their outputs,
and apply an output projection. Work through the dimensions before coding.
Then connect attention outputs to next-token prediction and shifted targets.

## Source

[Attention Is All You Need](https://arxiv.org/abs/1706.03762), especially
sections 3.2.1–3.2.3 and Figure 2. The paper introduces the formula that this
unit computes; the three-token experiments here are original toy examples.
