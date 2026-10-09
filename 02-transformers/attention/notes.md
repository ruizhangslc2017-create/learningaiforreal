# Attention: scores, weights, and a weighted sum

Session: 2026-10-09

Goal: trace a single attention head on three token vectors and reconstruct
the calculation in Python.

## The calculation

A query compares with each key. Softmax converts those scores into weights;
the output combines the corresponding values using those weights.

For query position `i` and key position `j`:

```text
score[i, j] = dot(query[i], key[j]) / sqrt(d_k)
weight[i, j] = exp(score[i, j]) / sum_j exp(score[i, j])
output[i] = sum_j weight[i, j] * value[j]
```

The matrix form is:

```text
S = Q @ K.T / sqrt(d_k)
A = softmax(S, across each row)
O = A @ V
```

For self-attention on `n` tokens, `Q` and `K` have shape `(n, d_k)`,
`V` has shape `(n, d_v)`, `S` and `A` have shape `(n, n)`, and `O` has
shape `(n, d_v)`. Row `i` says how much token `i` takes from each token `j`.

Each row of `A` sums to 1. The output is a weighted average of **values**;
the keys determine the scores.

Source for the attention formula and scaling:
[Attention Is All You Need, section 3.2.1](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1).

## Worked example: three tokens

Use labels `I`, `like`, and `tea`, with deliberately invented vectors:

```text
X = [[1, 0],    # I
     [0, 1],    # like
     [1, 1]]    # tea

Q = K = V = X
d_k = 2
```

The labels help track rows; these vectors carry no learned language meaning.
Using `Q = K = V = X` corresponds to identity projections. Learned
projections can come in a later session.

First compute all pairwise dot products, then scale:

```text
Q @ K.T = [[1, 0, 1],
           [0, 1, 1],
           [1, 1, 2]]

S = [[0.707107, 0.000000, 0.707107],
     [0.000000, 0.707107, 0.707107],
     [0.707107, 0.707107, 1.414214]]
```

For the first row, `exp(0.707107)` is about `2.028115`:

```text
unnormalized = [2.028115, 1.000000, 2.028115]
sum = 5.056230
weights = [0.401112, 0.197776, 0.401112]

output[0] = 0.401112 * [1, 0]
          + 0.197776 * [0, 1]
          + 0.401112 * [1, 1]
          = [0.802224, 0.598888]
```

The complete result, rounded to six decimal places:

```text
A = [[0.401112, 0.197776, 0.401112],
     [0.197776, 0.401112, 0.401112],
     [0.248255, 0.248255, 0.503490]]

O = [[0.802224, 0.598888],
     [0.598888, 0.802224],
     [0.751745, 0.751745]]
```

## Numerical detail: stable softmax

Subtract the largest score before taking exponentials:

```text
exp(score[j] - max_score) / sum_j exp(score[j] - max_score)
```

This preserves the ratios because every exponential receives the same
factor, `exp(-max_score)`. It also avoids overflowing on large finite scores.

## Run and inspect

From this folder, run:

```bash
python3 implementation.py
```

The program prints scores, weights, and outputs for both unmasked and causal
attention. Its checks verify normalization, the worked output, zero future
weights, and isolation from a changed future token.

Try changing only the third **value** vector to `[10, 10]`, leaving queries
and keys fixed. Predict what stays the same before running it again.

## Causal masking: prevent future information

At position `i`, allow values from positions `j <= i`. Before softmax,
replace future scores with negative infinity. Its exponential is zero:

```text
allowed = [[yes, no,  no ],
           [yes, yes, no ],
           [yes, yes, yes]]

masked S = [[0.707107,     -inf,     -inf],
            [0.000000, 0.707107,     -inf],
            [0.707107, 0.707107, 1.414214]]

causal A = [[1.000000, 0.000000, 0.000000],
            [0.330238, 0.669762, 0.000000],
            [0.248255, 0.248255, 0.503490]]

causal O = [[1.000000, 0.000000],
            [0.330238, 0.669762],
            [0.751745, 0.751745]]
```

The first output must equal the first value: there is only one allowed source.
The final row matches unmasked attention because it has no future positions.

Change the third input from `[1, 1]` to `[10, 10]`, and recompute all of
`Q`, `K`, and `V`. The first two causal outputs stay the same. Their allowed
queries, keys, and values have not changed. The unmasked outputs do change.

### A mistake to reproduce

Compute softmax first, then set future weights to zero. The first row becomes
`[0.401112, 0, 0]`, summing to `0.401112` instead of 1. Its output is too small.
Zeroing weights and explicitly renormalizing could recover the result, but
masking scores before softmax makes normalization part of one calculation.
Setting forbidden scores to **zero** also fails: `exp(0) = 1` still gives
future tokens positive weight.

### Why this connects to language modeling

For the sequence `I like tea`, next-token training can pair inputs `I, like`
with targets `like, tea`. The representation at input position `i` predicts
the token at `i + 1`. It may use its current token and earlier tokens. Allowing
the next input token would reveal the target. This is why the diagonal is
allowed while positions above it are masked.

The demo uses aligned, equal-length sequences. Cached decoding needs position
offsets, and cross-attention can use different query and key lengths; those
are outside this implementation's causal mode.

## Separate projections: Q, K, and V need different jobs

Use `project(X, W)` to multiply row vectors by a supplied weight matrix:

```text
Q = X @ W_Q        W_Q: (d_model, d_k)
K = X @ W_K        W_K: (d_model, d_k)
V = X @ W_V        W_V: (d_model, d_v)
```

Try fixed matrices `W_Q = [[1, 1], [0, 1]]`, `W_K = [[1, 0], [0, 1]]`,
and `W_V = [[2, 0], [0, 0.5]]`:

```text
Q = [[1, 1], [0, 1], [1, 2]]
K = [[1, 0], [0, 1], [1, 1]]
V = [[2, 0], [0, 0.5], [2, 0.5]]
```

Changing `W_Q` or `W_K` can change the weights. Changing only `W_V` changes
what gets combined while leaving weights unchanged. These matrices are
hand-chosen; a training loop would update them using gradients from a loss.
Run the projection experiment in the notebook and trace one row by hand.

## Why the square-root scale?

Under independent, zero-mean, unit-variance query and key components, each
product `q[d] * k[d]` has variance 1. Summing `d_k` independent products gives
variance `d_k`; dividing by `sqrt(d_k)` gives variance 1.

For example, scores `[-8, 0, 8]` give softmax approximately
`[0.000000, 0.000335, 0.999665]`. Dividing those scores by `sqrt(64) = 8`
gives `[-1, 0, 1]` and weights `[0.090031, 0.244728, 0.665241]`.
This controlled example illustrates concentration. It does not guarantee
uniform weights, and actual trained vectors need not satisfy the assumptions.

## Exercises and common confusions

1. Predict the second causal output, then calculate it from its two weights.
2. Make all three value vectors equal. Explain why every output is that value.
3. Permute the token order in unmasked attention. Check whether outputs permute
   with it. Explain why positional information would be needed to tell orders apart.
4. Change `W_V` to produce three output coordinates. Check the output shape.
5. Rebuild the attention function from its formula and compare it to this file.

Keep these distinctions clear:

- Attention weights refer to source positions; they are not vocabulary probabilities.
- Row sums are one; column sums need not be.
- Negative scores are valid. Softmax compares scores within a row.
- A masked row needs an allowed position; masking every score makes normalization undefined.

Record your own predictions and mistakes in [experiments.ipynb](experiments.ipynb).
Next: implement two heads, track their dimensions, and add an output projection.
